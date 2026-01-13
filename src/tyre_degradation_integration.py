import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from src.bayesian_tyre_model import BayesianTyreDegradationModel


class TyreDegradationIntegrator:

    
    def __init__(self, session=None, laps_df: Optional[pd.DataFrame] = None):
        self.session = session
        self._laps_df = laps_df
        self._model = BayesianTyreDegradationModel()
        self._initialized = False
        self._cache = {}
    
    def initialize_from_session(self) -> bool:
        """Initialize model from session data."""
        try:
            if self._laps_df is None:
                if self.session is None:
                    print("BayesianModel: No session or laps data provided")
                    return False
                self._laps_df = self.session.laps
            
            if self._laps_df is None or self._laps_df.empty:
                print("BayesianModel: Empty laps dataframe")
                return False
            
            print(f"BayesianModel: Fitting state-space model on {len(self._laps_df)} laps...")
            
            # Fit the model
            self._model.fit(self._laps_df)
            
            self._initialized = True
            
            # Print learned parameters
            print("BayesianModel: Degradation rates (seconds/lap):")
            for compound, rate in self._model.degradation_rates.items():
                print(f"  {compound}: {rate:.4f}")
            
            return True
            
        except Exception as e:
            print(f"BayesianModel initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def is_initialized(self) -> bool:
        """Check if model is initialized."""
        return self._initialized
    
    def get_tyre_health(
        self,
        driver_code: str,
        current_lap: int,
        force_refresh: bool = False
    ) -> Optional[Dict]:
        """Get tyre health - compatible with existing UI."""
        if not self._initialized:
            return None
        
        cache_key = f"{driver_code}_{current_lap}"
        if not force_refresh and cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            health_data = self._model.get_health(driver_code, current_lap, self._laps_df)
            
            if health_data:
                self._cache[cache_key] = health_data
            
            return health_data
            
        except Exception as e:
            print(f"BayesianModel query error for {driver_code} lap {current_lap}: {e}")
            return None
    
    def get_health_for_frame(
        self,
        driver_code: str,
        frame_data: Dict,
        frame_index: Optional[int] = None
    ) -> Optional[Dict]:
        """Get health from frame - compatible with existing UI."""
        if not frame_data or "drivers" not in frame_data:
            return None
        
        driver_pos = frame_data["drivers"].get(driver_code)
        if not driver_pos:
            return None
        
        lap = driver_pos.get("lap")
        if lap is None:
            return None
        
        try:
            lap_num = int(lap)
        except (ValueError, TypeError):
            return None
        
        return self.get_tyre_health(driver_code, lap_num)
    
    def clear_cache(self):
        """Clear cache."""
        self._cache.clear()



def format_tyre_health_bar(health: int, width: int = 100, height: int = 12) -> Dict:
    
    health = max(0, min(100, health))
    fill_width = (health / 100.0) * width
    
    if health >= 75:
        color = (0, 220, 0)
    elif health >= 50:
        ratio = (health - 50) / 25.0
        color = (int(220 * (1 - ratio)), 220, 0)
    elif health >= 25:
        ratio = (health - 25) / 25.0
        color = (220, int(220 * ratio), 0)
    else:
        ratio = health / 25.0
        color = (220, int(110 * ratio), 0)
    
    return {
        "width": width,
        "height": height,
        "fill_width": fill_width,
        "color": color,
        "health": health
    }


def format_degradation_text(health_data: Dict) -> str:
    
    if not health_data:
        return "N/A"
    
    compound = health_data.get("compound", "?")
    laps = health_data.get("laps_on_tyre", 0)
    health = health_data.get("health", 0)
    expected = health_data.get("expected_delta", 0.0)
    
    base = f"{compound} (L{laps}): {health}%"
    
    if expected > 0:
        base += f" • +{expected:.1f}s"
    
    if health_data.get("overdriving", False):
        base += " ⚠"
    
    
    if "uncertainty" in health_data:
        base += f" (±{health_data['uncertainty']:.2f}s)"
    
    return base