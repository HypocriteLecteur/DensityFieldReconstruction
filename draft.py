import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ============================================================
# Parameters (change these)
# ============================================================

f = 3200.0      # focal length (pixels)
dx = 1.0        # measurement error (pixels)
E = 0.50        # desired position error (meters)

y_min = 20.0     # minimum target distance (meters)
y_max = 200.0   # maximum target distance (meters)
num_points = 300

ys = np.linspace(y_min, y_max, num_points)

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(ys, ys**2/f/E)

plt.xlabel("Target distance d (m)")
plt.ylabel("Required baseline b (m)")
plt.title(
    f"Required baseline\n"
    f"f={f:.0f} px, $\delta$s=1 px, $\delta$d={E:.3f} m"
)
plt.grid(True)
plt.tight_layout()

bs = np.linspace(1, 25, num_points)

plt.figure(figsize=(8, 5))
plt.plot(bs, 100**2/bs/3200)

plt.xlabel("baseline b (m)")
plt.ylabel("depth error $\delta$d (m)")
plt.title(
    f"f={f:.0f} px, $\delta$s=1 px, Target distance d=100 m"
)
plt.grid(True)
plt.tight_layout()

ds = np.linspace(20, 300, num_points)

plt.figure(figsize=(8, 5))
plt.plot(ds, 20/ds*0.01*180/np.pi)

plt.xlabel("Target distance d (m)")
plt.ylabel("angle accuracy $\delta$$\theta$ (deg)")
plt.title(
    f"f={f:.0f} px, $\delta$d=1% d, Baseline b=20 m"
)

plt.grid(True)
plt.tight_layout()
plt.show()