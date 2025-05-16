import time
import signal
import numpy as np
import matplotlib.pyplot as plt
from eyeware.beam_eye_tracker import (
    API, ViewportGeometry, Point, TrackingConfidence
)


# --- CONFIGURAÇÃO ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1600, 900
SAMPLE_RATE     = 5.0
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE

# --- Inicializa o Beam ---
viewport = ViewportGeometry(Point(0, 0), Point(SCREEN_WIDTH, SCREEN_HEIGHT))
api = API("MinhaApp", viewport)
api.attempt_starting_the_beam_eye_tracker()
print("Tentando conectar com o Beam Eye Tracker…")

# handler para Ctrl+C
signal.signal(signal.SIGINT,
              lambda *a: (_ for _ in ()).throw(KeyboardInterrupt))

# coletores
timestamps = []
states     = []

try:
    while True:
        st   = api.get_latest_tracking_state_set()
        user = st.user_state()
        gaze = user.unified_screen_gaze

        # -- ponto normalizado (0–1)
        norm = gaze.normalized_point_of_regard

        # 1) Testa se saiu da área visível
        out_of_bounds = (
            norm.x < 0.0 or norm.x > 1.0 or
            norm.y < 0.0 or norm.y > 1.0
        )

        # 2) lost → desviado; caso contrário → olhando
        looking = (
            gaze.confidence != TrackingConfidence.LOST_TRACKING and
            not out_of_bounds
        )

        # armazena
        t = time.time()
        timestamps.append(t)
        states.append(looking)

        # log
        print(f"[{time.strftime('%H:%M:%S', time.localtime(t))}] "
              f"Olhando? {'Sim' if looking else 'Não'} | "
              f"norm=({norm.x:.2f},{norm.y:.2f}) conf={gaze.confidence}")

        time.sleep(SAMPLE_INTERVAL)

except KeyboardInterrupt:
    print("\nCaptura finalizada. Gerando gráfico…")

# --- PLOTAGEM ---

# Normaliza tempo
t0    = timestamps[0]
times = np.array(timestamps) - t0

# Cores para o plot
colors = ['red' if s else 'green' for s in states]

fig, axes = plt.subplots(2, 1,
                         gridspec_kw={'height_ratios': [1, 0.3]},
                         sharex=True, figsize=(10, 4))

axes[0].scatter(times, [1]*len(times), c=colors, s=50)
axes[0].set_yticks([]); axes[0].set_ylabel("Olhar")
axes[0].set_title("Vermelho = olhando · Verde = desviou")

for t, c in zip(times, colors):
    axes[1].barh(0, SAMPLE_INTERVAL, left=t, color=c)
axes[1].set_yticks([]); axes[1].set_xlabel("Tempo (s)")
axes[1].set_xlim(0, times[-1] + SAMPLE_INTERVAL)

plt.tight_layout(); plt.show()