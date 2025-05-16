"""
Logger – compatível com beam-eye-tracker ≥ 2.0
Captura head-pose + gaze do Beam Eye Tracker Demo e salva em data1.csv.
Dependências: pip install beam-eye-tracker numpy pandas
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

from eyeware.beam_eye_tracker import (
    API,
    ViewportGeometry,
    Point,
    TrackingConfidence,
)

# ----------------------- CONFIGURAÇÃO RÁPIDA ------------------------
VIEWPORT_SIZE_PX = (1600, 900)   # <— troque para a resolução do seu monitor
CSV_PATH         = Path("data1.csv")
LOOP_PERIOD_S    = 1 / 5          # 5 Hz (use 1/30 p/ 30 Hz)
# -------------------------------------------------------------------

viewport = ViewportGeometry(Point(0, 0), Point(*VIEWPORT_SIZE_PX))
tracker  = API("BeamLogger", viewport)
tracker.attempt_starting_the_beam_eye_tracker()

df = pd.DataFrame()
print("=== Logger iniciado (Ctrl+C para sair) ===")

try:
    while True:
        # snapshot
        state_set  = tracker.get_latest_tracking_state_set()
        user_state = state_set.user_state()

        head_pose   = user_state.head_pose
        screen_gaze = user_state.unified_screen_gaze

        head_lost = head_pose.confidence == TrackingConfidence.LOST_TRACKING
        gaze_lost = screen_gaze.confidence == TrackingConfidence.LOST_TRACKING

        # ---- impressão (opcional) ----
        print("Head lost:", head_lost,
              "| Gaze lost:", gaze_lost,
              "| x:", int(screen_gaze.point_of_regard.x),
              "| y:", int(screen_gaze.point_of_regard.y),
              "| conf:", screen_gaze.confidence)

        # ---- guarda tudo num DataFrame ----
        now = time.strftime("%b,%d,%H,%Y", time.localtime()).split(",")
        row = {
            "gaze_X": int(screen_gaze.point_of_regard.x),
            "gaze_Y": int(screen_gaze.point_of_regard.y),
            "mes":    now[0],
            "dia":    now[1],
            "hora":   now[2],
            "ano":    now[3],
            "olhando_tela": str(not (head_lost or gaze_lost)),
            "confidence":   int(screen_gaze.confidence), 
        }

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)

        time.sleep(LOOP_PERIOD_S)

except KeyboardInterrupt:
    print("\nColeta encerrada pelo usuário.")
      