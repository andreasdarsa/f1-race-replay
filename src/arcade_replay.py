import os
import arcade
from src.interfaces.race_replay import F1RaceReplayWindow

def run_arcade_replay(frames, track_statuses, example_lap, drivers, title,
                      playback_speed=1.0, driver_colors=None, circuit_rotation=0.0, total_laps=None,
                      visible_hud=True, ready_file=None, session=None):
    window = F1RaceReplayWindow(
    frames=frames,
    track_statuses=track_statuses,
    example_lap=example_lap,
    drivers=drivers,
    title=title,
    playback_speed=playback_speed,
    driver_colors=driver_colors,
    circuit_rotation=circuit_rotation,
    left_ui_margin=340,
    right_ui_margin=260,
    total_laps=total_laps,
    visible_hud=visible_hud,
    session=session  
    )
    # Signal readiness to parent process (if requested) after window created
    if ready_file:
        try:
            with open(ready_file, 'w') as f:
                f.write('ready')
        except Exception:
            pass
    arcade.run()