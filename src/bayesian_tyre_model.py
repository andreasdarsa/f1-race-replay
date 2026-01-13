import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize


@dataclass
class StateSpaceConfig:
    """Configuration for state-space model parameters."""

    sigma_epsilon: float = 0.3
    
    
    sigma_eta: float = 0.1
    
    
    fuel_effect_prior: float = 0.032
    
    
    starting_fuel: float = 110.0
    
    
    fuel_burn_rate: float = 1.6
    
    
    degradation_rate_hard: float = 0.01
    degradation_rate_medium: float = 0.03
    degradation_rate_soft: float = 0.05
    
    reset_pace_hard: float = 69.5
    reset_pace_medium: float = 69.0
    reset_pace_soft: float = 68.5
    
    
    enable_warmup: bool = True
    warmup_laps: int = 3


class BayesianTyreDegradationModel:
    """
    Bayesian state-space model for tire degradation.
    
    State equation:
        α_{t+1} = (1 - I_pit) * (α_t + ν) + I_pit * α_reset + η_t
        
    Observation equation:
        y_t = α_t + γ * fuel_t + ε_t
        
    Where:
        α_t = latent tire pace (true lap time capability)
        ν = degradation rate (seconds per lap)
        γ = fuel effect (seconds per kg)
        I_pit = indicator for pit stop
        α_reset = pace reset after pit stop
    """
    
    def __init__(self, config: Optional[StateSpaceConfig] = None):
        self.config = config or StateSpaceConfig()
        
        
        self.degradation_rates = {
            'HARD': self.config.degradation_rate_hard,
            'MEDIUM': self.config.degradation_rate_medium,
            'SOFT': self.config.degradation_rate_soft
        }
        
        self.reset_paces = {
            'HARD': self.config.reset_pace_hard,
            'MEDIUM': self.config.reset_pace_medium,
            'SOFT': self.config.reset_pace_soft
        }
        
        self.fuel_effect = self.config.fuel_effect_prior
        self.sigma_epsilon = self.config.sigma_epsilon
        self.sigma_eta = self.config.sigma_eta
        
        
        self._latent_states = {}  # driver -> List[alpha_t]
        self._fitted = False
        
    def fit(self, laps_df: pd.DataFrame, driver: Optional[str] = None):
        if driver:
            laps_df = laps_df[laps_df['Driver'] == driver]
        
        
        laps_clean = self._prepare_data(laps_df)
        
        self._estimate_parameters(laps_clean)
        
        
        self._compute_latent_states(laps_clean)
        
        self._fitted = True
        
    def _prepare_data(self, laps_df: pd.DataFrame) -> pd.DataFrame:
        
        laps = laps_df.copy()
        
        
        is_pit_out = laps["PitOutTime"].notna()
        is_pit_in = laps["PitInTime"].notna()
        
        laps = laps[
            (laps["LapNumber"] > 1) &  # Exclude lap 1
            ~is_pit_in &
            ~is_pit_out &
            laps["LapTime"].notna() &
            laps["Compound"].notna()
        ]
        
        laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
        
        
        laps["FuelMass"] = (
            self.config.starting_fuel -
            (laps["LapNumber"] - 1) * self.config.fuel_burn_rate
        ).clip(lower=0)
        
        
        laps = laps.sort_values(["Driver", "LapNumber"])
        
        return laps
    
    def _estimate_parameters(self, laps_df: pd.DataFrame):
        # Group by compound and estimate degradation rates
        for compound in ['HARD', 'MEDIUM', 'SOFT']:
            compound_laps = laps_df[laps_df['Compound'] == compound]
            
            if len(compound_laps) < 5:
                continue
            for driver in compound_laps['Driver'].unique():
                driver_laps = compound_laps[compound_laps['Driver'] == driver]
                
                for stint in driver_laps['Stint'].unique():
                    stint_laps = driver_laps[driver_laps['Stint'] == stint]
                    
                    if len(stint_laps) < 5:
                        continue
                    
                    
                    stint_laps = stint_laps.copy()
                    stint_laps['LapOnTyre'] = range(1, len(stint_laps) + 1)
                    

                    first_lap_time = stint_laps.iloc[0]['LapTimeSeconds']
                    fuel_corrected = (
                        stint_laps['LapTimeSeconds']
                        - self.fuel_effect * stint_laps['FuelMass']
                        )

                    first_fc = fuel_corrected.iloc[0]
                    stint_laps['DeltaFromFirst'] = fuel_corrected - first_fc

                    
                    
                    if self.config.enable_warmup:
                        warmup_laps = self.config.warmup_laps
                        analysis_laps = stint_laps[stint_laps['LapOnTyre'] > warmup_laps]
                    else:
                        analysis_laps = stint_laps
                    
                    if len(analysis_laps) > 2:
                        # Linear fit: delta = ν * lap_on_tyre
                        x = analysis_laps['LapOnTyre'].values
                        y = analysis_laps['DeltaFromFirst'].values
                        
                        # Robust linear regression
                        if len(x) > 0 and np.std(y) > 0:
                            slope = np.polyfit(x, y, 1)[0]
                            
                            # Update degradation rate (weighted average with prior)
                            prior_weight = 0.6
                            self.degradation_rates[compound] = (
                                prior_weight * self.degradation_rates[compound] +
                                (1 - prior_weight) * max(0, slope)
                            )
        if(
            "HARD" in self.degradation_rates
            and "MEDIUM" in self.degradation_rates
        ):
            self.degradation_rates["HARD"]=min(
                self.degradation_rates["HARD"], 0.6* self.degradation_rates["MEDIUM"]
            )
    def _compute_latent_states(self, laps_df: pd.DataFrame):
        self._latent_states = {}
        self._latent_uncertainty = {}

        obs_var = self.sigma_epsilon ** 2
        proc_var = self.sigma_eta ** 2

        for driver in laps_df["Driver"].unique():
            driver_laps = (
                laps_df[laps_df["Driver"] == driver]
                .sort_values("LapNumber")
                .reset_index(drop=True)
            )

            mu_alpha = None
            var_alpha = None

            states = []
            variances = []

            prev_stint = None

            for _, lap in driver_laps.iterrows():
                compound = lap["Compound"]
                lap_time = lap["LapTimeSeconds"]
                fuel = lap["FuelMass"]
                stint = lap["Stint"]

                if mu_alpha is None or stint != prev_stint:
                    mu_alpha = self.reset_paces[compound]
                    var_alpha = proc_var
                    prev_stint = stint
                else:
                    nu = self.degradation_rates[compound]
                    mu_pred = mu_alpha + nu
                    var_pred = var_alpha + proc_var

                    expected_lap = mu_pred + self.fuel_effect * fuel
                    innovation = lap_time - expected_lap

                    innovation_var = var_pred + obs_var
                    kalman_gain = var_pred / innovation_var
                    
                    effective_nu=nu*(1-kalman_gain)

                    mu_alpha = mu_pred + kalman_gain * innovation
                    var_alpha = (1.0 - kalman_gain) * var_pred

                states.append(mu_alpha)
                variances.append(var_alpha)

            self._latent_states[driver] = states
            self._latent_uncertainty[driver] = variances

    
    def predict_next_lap(
        self,
        driver: str,
        current_lap: int,
        laps_df: pd.DataFrame
    ) -> Tuple[float, float, Dict]:

        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Get driver's laps up to current lap
        driver_laps = laps_df[
            (laps_df['Driver'] == driver) &
            (laps_df['LapNumber'] <= current_lap)
        ].sort_values('LapNumber')
        
        if driver_laps.empty:
            return None, None, {}
        
        
        last_lap = driver_laps.iloc[-1]
        compound = last_lap['Compound']
        stint = last_lap['Stint']
        
        
        stint_laps = driver_laps[driver_laps['Stint'] == stint]
        laps_on_tyre = len(stint_laps)
        
        
        if laps_on_tyre == 1:
            
            alpha_t = self.reset_paces[compound]
        else:
            
            alpha_t = self.reset_paces[compound] + (laps_on_tyre - 1) * self.degradation_rates[compound]
        
        
        if self.config.enable_warmup and laps_on_tyre <= self.config.warmup_laps:
            warmup_penalty = self._compute_warmup_penalty(compound, laps_on_tyre)
            alpha_t = alpha_t + warmup_penalty
        
        
        next_lap = current_lap + 1
        fuel_next = max(0, self.config.starting_fuel - (next_lap - 1) * self.config.fuel_burn_rate)
        
        
        predicted_time = alpha_t + self.fuel_effect * fuel_next
        
        
        var_alpha = self._latent_uncertainty[driver][-1]
        std_dev = np.sqrt(var_alpha + self.sigma_epsilon ** 2)
        
        
        max_laps = self._estimate_max_laps(compound)
        health = max(0, min(100, 100 * (1 - laps_on_tyre / max_laps)))
        
        info = {
            'latent_pace': alpha_t,
            'predicted_time': predicted_time,
            'std_dev': std_dev,
            'health': int(health),
            'laps_on_tyre': laps_on_tyre,
            'compound': compound,
            'degradation_rate': self.degradation_rates[compound],
            'confidence_95': (predicted_time - 1.96 * std_dev, predicted_time + 1.96 * std_dev)
        }
        
        return predicted_time, std_dev, info
    
    def _compute_warmup_penalty(self, compound: str, lap_on_tyre: int) -> float:
        if lap_on_tyre > self.config.warmup_laps:
            return 0.0
        
        
        warmup_penalties = {
            'HARD': -0.3,
            'MEDIUM': -0.2,
            'SOFT': -0.1
        }
        
        max_penalty = warmup_penalties.get(compound, -0.2)
        
        
        penalty = max_penalty * (1 - (lap_on_tyre - 1) / self.config.warmup_laps)
        
        return penalty
    
    def _estimate_max_laps(self, compound: str) -> float:

        max_degradation = 2.0
        rate = self.degradation_rates.get(compound, 0.05)
        
        if rate <= 0:
            return 50  # Fallback
        
        return max_degradation / rate
    
    def get_degradation_rate(self, compound: str) -> float:

        return self.degradation_rates.get(compound, 0.05)
    
    def get_health(
        self,
        driver: str,
        current_lap: int,
        laps_df: pd.DataFrame
    ) -> Dict:
        _, _, info = self.predict_next_lap(driver, current_lap, laps_df)
        
        if not info:
            return None
        
        return {
            'compound': info['compound'],
            'laps_on_tyre': info['laps_on_tyre'],
            'health': info['health'],
            'expected_delta': info['degradation_rate'] * info['laps_on_tyre'],
            'actual_delta': 0.0,  
            'overdriving': False,
            'uncertainty': info['std_dev'],
            'latent_pace': info['latent_pace']
        }


