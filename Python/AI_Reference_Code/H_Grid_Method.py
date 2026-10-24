import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# AIRFOIL GEOMETRY CLASS (As provided by user)
# =====================================================================
class NACA4DigitAirfoil:
    def __init__(self, code: str, chord: float, precision: int):
        self.code = code
        self.c = chord
        self.precision = precision
        self.parameters()
        self.t = np.linspace(0, self.c, self.precision, dtype=float)
        self.compute_geometry()
    
    def parameters(self):
        NACA_n = [float(x) for x in self.code]
        self.m = (NACA_n[0] * 0.01)
        self.p = (NACA_n[1] * 0.1)
        self.T = (((NACA_n[2] * 10) + NACA_n[3]) * 0.01)
    
    def camber_line(self, t):
        if self.m == 0 or self.p == 0:
            return np.zeros_like(t, dtype=float)
        t_norm = t / self.c
        con_list = [(t >= 0) & (t <= self.p * self.c), (t >= self.p * self.c) & (t <= self.c)]
        func_list = [(self.m/self.p**2)*((2*self.p*t_norm)-(t_norm**2)),
                     (self.m/((1- self.p)**2))*((1-(2*self.p))+(2*self.p*t_norm)-(t_norm**2))]
        return np.select(con_list, func_list, 0.0)
    
    def compute_geometry(self):
        self.y_c = self.camber_line(self.t)
        dy_c = np.gradient(self.y_c, self.t)
        Theta_d = np.arctan(dy_c)
        
        t_n = self.t / self.c
        y_t = 5*(self.T * self.c)*((0.2969*np.sqrt(t_n)) -
                   (0.1260*t_n) -
                   (0.3516*(t_n**2)) +
                   (0.2843*(t_n**3)) -
                   (0.1015*(t_n**4)))
        
        self.X_u = (self.t - (y_t * np.sin(Theta_d)))
        self.Y_u = self.y_c + (y_t * np.cos(Theta_d))
        self.X_l = (self.t + (y_t * np.sin(Theta_d)))
        self.Y_l = self.y_c - (y_t * np.cos(Theta_d))

# =====================================================================
# ELLIPTIC H-GRID GENERATOR
# =====================================================================
def generate_elliptic_h_grid(airfoil, Ni=91, Nj=41, tunnel_height=3.0, tunnel_length=5.0):
    """
    Generates a structured curvilinear H-grid around an airfoil inside a wind tunnel
    using an Elliptic PDE smoothing scheme.
    """
    # Grid dimensions allocations
    # Airfoil profile is centered axially in the domain
    x = np.zeros((Ni, Nj))
    y = np.zeros((Ni, Nj))
    
    # Define block indices along the longitudinal direction (i-axis)
    # The airfoil surface matches the center section of the domain index split
    i_airfoil_points = airfoil.precision
    i_start = (Ni - i_airfoil_points) // 2
    i_end = i_start + i_airfoil_points - 1
    
    # Establish the constant splitting line location along the vertical axis (j-axis)
    j_mid = Nj // 2 
    
    # Wind tunnel dimensions boundaries
    x_inflow = -tunnel_length * 0.4
    x_outflow = tunnel_length * 0.6
    y_bottom = -tunnel_height / 2.0
    y_top = tunnel_height / 2.0
    
    # Step 1: Initialize Boundaries (Dirichlet Conditions)
    # Inflow and Outflow
    for j in range(Nj):
        eta_frac = j / (Nj - 1)
        y_val = y_bottom + eta_frac * (y_top - y_bottom)
        x[0, j] = x_inflow
        y[0, j] = y_val
        x[-1, j] = x_outflow
        y[-1, j] = y_val
        
    # Top and Bottom Tunnel Walls
    for i in range(Ni):
        xi_frac = i / (Ni - 1)
        x_val = x_inflow + xi_frac * (x_outflow - x_inflow)
        x[i, 0] = x_val
        y[i, 0] = y_bottom
        x[i, -1] = x_val
        y[i, -1] = y_top

    # Step 2: Establish the Seam/Airfoil Line (Internal Dirichlet & Slit line)
    # Ensure matching points at the leading edge and trailing edge
    x_airfoil_start = 0.0
    x_airfoil_end = airfoil.c
    
    # Left wake line (Inflow to Leading Edge)
    for i in range(0, i_start):
        frac = i / i_start
        x[i, j_mid] = x_inflow + frac * (x_airfoil_start - x_inflow)
        y[i, j_mid] = 0.0  # Centerline alignment
        
    # Right wake line (Trailing Edge to Outflow)
    for i in range(i_end + 1, Ni):
        frac = (i - i_end) / (Ni - 1 - i_end)
        x[i, j_mid] = x_airfoil_end + frac * (x_outflow - x_airfoil_end)
        y[i, j_mid] = 0.0  # Centerline alignment

    # Step 3: Interpolate internal fields algebraically (TFI initialization)
    # Provides a baseline non-collapsed mesh structure before solving the PDEs
    for i in range(Ni):
        for j in range(Nj):
            # Bi-linear blending field interpolation logic between inner seam and walls
            if j < j_mid:  # Lower half domain
                frac = j / j_mid
                # Interpolating between bottom boundary (j=0) and center line boundary (j=j_mid)
                if i_start <= i <= i_end:
                    # Point falls exactly on lower airfoil profile surface
                    x_target = airfoil.X_l[i - i_start]
                    y_target = airfoil.Y_l[i - i_start]
                else:
                    x_target = x[i, j_mid]
                    y_target = y[i, j_mid]
                x[i, j] = x[i, 0] + frac * (x_target - x[i, 0])
                y[i, j] = y[i, 0] + frac * (y_target - y[i, 0])
            elif j > j_mid:  # Upper half domain
                frac = (j - j_mid) / (Nj - 1 - j_mid)
                # Interpolating between center line boundary (j=j_mid) and top wall (j=Nj-1)
                if i_start <= i <= i_end:
                    # Point falls exactly on upper airfoil profile surface
                    x_target = airfoil.X_u[i - i_start]
                    y_target = airfoil.Y_u[i - i_start]
                else:
                    x_target = x[i, j_mid]
                    y_target = y[i, j_mid]
                x[i, j] = x_target + frac * (x[i, -1] - x_target)
                y[i, j] = y_target + frac * (y[i, -1] - y_target)

    # Re-inject strict airfoil surface coordinate bounds to avoid interpolation drift
    x[i_start:i_end+1, j_mid] = 0.5 * (airfoil.X_u + airfoil.X_l)
    y[i_start:i_end+1, j_mid] = 0.5 * (airfoil.Y_u + airfoil.Y_l)

    # Step 4: Iterative Elliptic PDE Relaxation Solver (Gauss-Seidel Execution)
    max_iter = 300
    w = 1.0  # Relaxation factor parameter
    
    for iteration in range(max_iter):
        # Arrays to hold updates
        x_new = x.copy()
        y_new = y.copy()
        
        # Loop over interior grid coordinates fields
        for i in range(1, Ni - 1):
            for j in range(1, Nj - 1):
                # Skip updates on the airfoil boundary wall elements directly
                if j == j_mid and (i_start <= i <= i_end):
                    continue
                    
                # Compute computational grid metrics via finite difference discretization
                x_xi  = 0.5 * (x[i+1, j] - x[i-1, j])
                x_eta = 0.5 * (x[i, j+1] - x[i, j-1])
                y_xi  = 0.5 * (y[i+1, j] - y[i-1, j])
                y_eta = 0.5 * (y[i, j+1] - y[i, j-1])
                
                # Elliptic Coordinate transformation metrics
                alpha = x_eta**2 + y_eta**2
                beta  = x_xi * x_eta + y_xi * y_eta
                gamma = x_xi**2 + y_xi**2
                
                # Discretized central second derivatives components
                x_vivi = x[i+1, j] + x_new[i-1, j]
                x_eteta = x[i, j+1] + x_new[i, j-1]
                x_vieta = 0.25 * (x[i+1, j+1] - x[i-1, j+1] - x[i+1, j-1] + x[i-1, j-1])
                
                y_vivi = y[i+1, j] + y_new[i-1, j]
                y_eteta = y[i, j+1] + y_new[i, j-1]
                y_vieta = 0.25 * (y[i+1, j+1] - y[i-1, j+1] - y[i+1, j-1] + y[i-1, j-1])
                
                # Implicit Quasilinear Elliptic Form execution update
                denom = 2.0 * (alpha + gamma)
                
                x_res = (alpha * x_vivi - 2.0 * beta * x_vieta + gamma * x_eteta) / denom
                y_res = (alpha * y_vivi - 2.0 * beta * y_vieta + gamma * y_eteta) / denom
                
                x_new[i, j] = x[i, j] + w * (x_res - x[i, j])
                y_new[i, j] = y[i, j] + w * (y_res - y[i, j])
                
        x = x_new
        y = y_new

    # Step 5: Enforce asymmetric physical split boundary condition properties onto the Airfoil Slit Seam
    # This matches upper J metrics to Upper coordinates, and lower J metrics to Lower coordinates
    for i in range(i_start, i_end + 1):
        idx = i - i_start
        # Upper face matching mapping
        x[i, j_mid] = airfoil.X_u[idx]
        y[i, j_mid] = airfoil.Y_u[idx]
        
    return x, y, i_start, i_end, j_mid

# =====================================================================
# INSTANTIATION AND SCENARIO PLOTTING
# =====================================================================
chord_length =  1.0
precision_points = 31 # Airfoil points profile discretization density
airfoil = NACA4DigitAirfoil(code="2412", chord=chord_length, precision=precision_points)

# Run H-Grid System Generator Engine
X, Y, i_s, i_e, j_m = generate_elliptic_h_grid(airfoil, Ni=81, Nj=36, tunnel_height=2.5, tunnel_length=4.5)

# Visualizing Grid Domain
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_aspect('equal')

# Plot constant-xi grid lines (i-indices lines)
for i in range(X.shape[0]):
    ax.plot(X[i, :], Y[i, :], color='gray', linewidth=0.4, alpha=0.7)

# Plot constant-eta grid lines (j-indices lines)
for j in range(X.shape[1]):
    if j == j_m:
        # Highlight the splitting centerline wake paths safely
        ax.plot(X[:i_s+1, j], Y[:i_s+1, j], color='blue', linewidth=1.2, label='Wake Centerline Seam')
        ax.plot(X[i_e:, j], Y[i_e:, j], color='blue', linewidth=1.2)
    else:
        ax.plot(X[:, j], Y[:, j], color='gray', linewidth=0.4, alpha=0.7)

# Overlay physical solid airfoil profile surface lines cleanly
ax.plot(airfoil.X_u, airfoil.Y_u, 'r-', linewidth=2.0, label='Airfoil Upper Profile')
ax.plot(airfoil.X_l, airfoil.Y_l, 'black', linewidth=2.0, label='Airfoil Lower Profile')
ax.set_title("Elliptic Smooth Structured H-Grid Domain Map (Wind Tunnel Framework Scenario)", fontsize=12, fontweight='bold')
ax.set_xlabel("Physical Axis: X [m]", fontsize=10)
ax.set_ylabel("Physical Axis: Y [m]", fontsize=10)
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()