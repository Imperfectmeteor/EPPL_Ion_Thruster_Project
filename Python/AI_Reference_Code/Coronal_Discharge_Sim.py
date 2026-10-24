import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Numerical Grid & Coordinate Domain (in mm)
nx, ny = 100, 100
x = np.linspace(-15, 15, nx)   # Horizontal span: -15mm to +15mm
y = np.linspace(-10, 0, ny)    # Vertical span: -10mm (Plate) to 0mm (Pin Tip)
X, Y = np.meshgrid(x, y)

# 2. Physics & Boundary Conditions Definition
V_app = -10000.0  # Applied High Voltage (-10 kV) at Pin Cathode
eps0 = 8.854e-12  # Permittivity of free air

# Initialize arrays
Phi = np.zeros((ny, nx))
rho = np.zeros((ny, nx))

# 3. Model Space Fields (Analytical Steady-State Continuum)
for i in range(ny):
    for j in range(nx):
        dx = x[j]
        dy = y[i] - 0.15  # Shift slightly to prevent singularity division by zero
        r = np.sqrt(dx**2 + dy**2)
        
        # Logarithmic potential drop-off mirroring needle geometry
        if r < 12:
            Phi[i, j] = V_app * (np.log(12 / r) / np.log(12 / 0.15))
        else:
            Phi[i, j] = 0
            
        # Unipolar charge density distribution: heavily concentrated at needle tip (0,0)
        rho[i, j] = 6.5e-5 * (0.3 / (r + 0.3)) ** 2.2

# Enforce strict plate boundary condition (Grounded Anode at y = -10mm)
Phi[0, :] = 0  

# 4. Calculate Electric Field E = -∇Φ (Converting mm grid spacing back to Meters)
dy_meters = (y[1] - y[0]) * 1e-3
dx_meters = (x[1] - x[0]) * 1e-3
Ey, Ex = np.gradient(-Phi, dy_meters, dx_meters)

# 5. Compute the Volumetric Navier-Stokes Source Term: f_L = rho * E
fx = rho * Ex
fy = rho * Ey
f_magnitude = np.sqrt(fx**2 + fy**2)

# 6. Render the Visual Spatial Map
plt.figure(figsize=(9, 7))

# Plot a continuous logarithmic heatmap representing force field intensity
force_contour = plt.contourf(X, Y, np.log10(f_magnitude + 1e-5), levels=100, cmap='inferno')

# Overlay normalized velocity/force vectors showing direction paths from pin to plate
# Using a grid-stride of 4 to prevent cluttering the canvas
stride = 4
plt.quiver(X[::stride, ::stride], Y[::stride, ::stride], 
           fx[::stride, ::stride]/f_magnitude[::stride, ::stride], 
           fy[::stride, ::stride]/f_magnitude[::stride, ::stride],
           color='white', alpha=0.4, scale=25, width=0.003, headwidth=4)

# Geometry Landmarks
plt.plot(0, 0, marker='v', color='cyan', markersize=10, label='Pin Tip (Cathode)')
plt.axhline(y=-10, color='lime', linewidth=3, label='Grounded Plate (Anode)')

# Polish Layout Elements
cbar = plt.colorbar(force_contour)
cbar.set_label('Volumetric Force Magnitude $\log_{10}(|\mathbf{f}_L|)$ [$\mathregular{N/m^3}$]', fontsize=11)
plt.title('EHD Electrostatic Source Term Spatial Vector Map ($\mathbf{f}_L = \\rho_e \mathbf{E}$)', fontsize=12, pad=15)
plt.xlabel('Horizontal Position [mm]', fontsize=10)
plt.ylabel('Vertical Position [mm]', fontsize=10)
plt.xlim(-15, 15)
plt.ylim(-10.5, 0.5)
plt.gca().set_aspect('equal')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.15)

plt.show()