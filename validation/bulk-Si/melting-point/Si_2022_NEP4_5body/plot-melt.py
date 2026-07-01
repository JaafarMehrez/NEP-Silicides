import numpy as np
import matplotlib.pyplot as plt

# Apply your specific theme settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman"],
    "text.usetex": True, 
    # FIX: Add amsmath to the preamble so \text{} works
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.top": True,
    "ytick.right": True,
})

# 1. Load the data
thermo = np.loadtxt('thermo.out')

# 2. Setup Time array
time = np.arange(1, len(thermo) + 1) / 5

# 3. Calculate mean of Temperature (second half)
half_idx = len(thermo) // 2
mean_temp = np.mean(thermo[half_idx:, 0])

# --- FIGURE 1: Temperature ---
plt.figure(figsize=(8, 6))

# Using an f-string with raw string r for LaTeX. 
# Note the double {{ }} around the unit if you use f-strings with LaTeX
temp_label = rf"$\langle T \rangle = {mean_temp:.2f} \text{{ K}}$"

plt.plot(time, thermo[:, 0], color='red', linewidth=1.5, label=temp_label)

plt.xlabel('Time (ps)')
plt.ylabel('Temperature (K)')

# Add legend: frameon=False removes the box
plt.legend(frameon=False, loc='best')

plt.tight_layout()
plt.savefig('temperature.png', dpi=300)
plt.show()

# --- FIGURE 2: Pressure ---
plt.figure(figsize=(8, 6))
plt.plot(time, thermo[:, 3], color='red',   label='$P_x$', linewidth=1.5)
plt.plot(time, thermo[:, 4], color='blue',  label='$P_y$', linewidth=1.5)
plt.plot(time, thermo[:, 5], color='black', label='$P_z$', linewidth=1.5)

plt.xlabel('Time (ps)')
plt.ylabel('Pressure (GPa)')
plt.legend(frameon=False, loc='best')
plt.tight_layout()
plt.savefig('pressure.png', dpi=300)
plt.show()
