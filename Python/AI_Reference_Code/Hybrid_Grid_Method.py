import numpy as np
import matplotlib.pyplot as plt
class NACA4DigitAirfoil:
    # Runs the class's functions and attaches inputs
    def __init__(self, code: str, chord: float, precision: int):
        self.code = code
        self.c = chord
        self.precision = precision
        self.parameters()
        self.t = np.linspace(0, self.c, self.precision, dtype=float)
        self.compute_geometry()
    
    # Defines the airfoil parameters from the NACA code
    def parameters(self):
        NACA_n = [float(x) for x in self.code]
        self.m = (NACA_n[0] * 0.01) # Maximum Camber
        self.p = (NACA_n[1] * 0.1)  # Position of Maximum Camber
        self.T = (((NACA_n[2] * 10) + NACA_n[3]) * 0.01) # Maximum Thickness
    
    # Computes the mean camber line for the given airfoil
    def camber_line(self, t):
        if self.m == 0 or self.p == 0:
            return np.zeros_like(t, dtype=float)
        
        t_norm = t / self.c
        con_list = [(t >= 0) & (t <= self.p * self.c), (t >= self.p * self.c) & (t <= self.c)]
        func_list = [(self.m/self.p**2)*((2*self.p*t_norm)-(t_norm**2)),
                     (self.m/((1- self.p)**2))*((1-(2*self.p))+(2*self.p*t_norm)-(t_norm**2))]
        return np.select(con_list, func_list, 0.0)
    
    # Computes the "physical" geometry of the airfoil
    def compute_geometry(self):
        self.y_c = self.camber_line(self.t)
        dy_c = np.gradient(self.y_c, self.t)
        Theta_d = np.arctan(dy_c) # Derivative Angle
        
        # Thickness distribution
        t_n = self.t / self.c # Normalizing t w/r c
        y_t = 5*(self.T * self.c)*((0.2969*np.sqrt(t_n)) -
                   (0.1260*t_n) -
                   (0.3516*(t_n**2)) +
                   (0.2843*(t_n**3)) -
                   (0.1015*(t_n**4)))
        
        # Upper and Lower Airfoil Geometric Functions
        self.X_u = (self.t - (y_t * np.sin(Theta_d)))
        self.Y_u = self.y_c + (y_t * np.cos(Theta_d))
        
        self.X_l = (self.t + (y_t * np.sin(Theta_d)))
        self.Y_l = self.y_c - (y_t * np.cos(Theta_d))

# ==============================================================================
# 1. GEOMETRY AND DIMENSIONAL SETUP
# ==============================================================================
chord_length = 1.0
precision_points = 50  
airfoil = NACA4DigitAirfoil(code="2412", chord=chord_length, precision=precision_points)

x_airfoil = np.concatenate([np.flip(airfoil.X_u), airfoil.X_l[1:]])
y_airfoil = np.concatenate([np.flip(airfoil.Y_u), airfoil.Y_l[1:]])

num_xi = len(x_airfoil)  # Points wrapped around the airfoil loop

# --- MULTI-BLOCK SIZE SPECIFICATIONS ---
num_eta_inner = 25       # Radial rings inside the high-orthogonality O-sleeve
num_eta_outer = 30       # Radial rows inside the rectangular H-grid transition zone
num_eta_total = num_eta_inner + num_eta_outer - 1 # Total merged radial resolution

# Define the physical boundaries of the outer wind tunnel box
x_min, x_max = -8.0 * chord_length, 8.0 * chord_length
y_min, y_max = -4.0 * chord_length, 4.0 * chord_length

# ==============================================================================
# 2. BLOCK 1 INNER BOUNDARY: GENERATING THE COMPACT O-SLEEVE PERIMETER
# ==============================================================================
# Smooth, local ellipse hovering closely around the airfoil profile
r_sleeve = 0.40 * chord_length
theta = np.linspace(0, 2 * np.pi, num_xi)
x_sleeve_outer = chord_length / 2.0 + (r_sleeve + 0.3 * chord_length) * np.cos(theta)
y_sleeve_outer = r_sleeve * np.sin(theta)

# Initialize Inner Coordinate Matrices
X_inner = np.zeros((num_eta_inner, num_xi))
Y_inner = np.zeros((num_eta_inner, num_xi))

# TFI Linear Initial Guess for the inner block sleeve
for j in range(num_eta_inner):
    w = j / (num_eta_inner - 1)
    X_inner[j, :] = (1.0 - w) * x_airfoil + w * x_sleeve_outer
    Y_inner[j, :] = (1.0 - w) * y_airfoil + w * y_sleeve_outer

# ==============================================================================
# 3. RUN THE HILGENSTOCK ELLIPTIC SMOOTHER (ONLY ON THE INNER SLEEVE BLOCK)
# ==============================================================================
iterations = 500
omega = 1.0
C_eta = slice(1, -1); L_eta = slice(0, -2); R_eta = slice(2, None)
i_idx = np.arange(num_xi); L_xi = np.roll(i_idx, 1); R_xi = np.roll(i_idx, -1)

delta_s = 0.005        # Exact target viscous cell spacing at the wall
decay_constant = 0.20  
decay_factors = np.exp(-decay_constant * np.arange(num_eta_inner))

for it in range(iterations):
    X_old = X_inner.copy()
    Y_old = Y_inner.copy()
    
    # Standard spatial metrics inside the inner block core
    x_xi  = 0.5 * (X_old[C_eta, R_xi] - X_old[C_eta, L_xi])
    y_xi  = 0.5 * (Y_old[C_eta, R_xi] - Y_old[C_eta, L_xi])
    x_eta = 0.5 * (X_old[R_eta, :] - X_old[L_eta, :])
    y_eta = 0.5 * (Y_old[R_eta, :] - Y_old[L_eta, :])
    
    alpha = x_eta**2 + y_eta**2
    beta  = x_xi * x_eta + y_xi * y_eta
    gamma = x_xi**2 + y_xi**2
    
    x_xi_xi   = X_old[C_eta, R_xi] - 2*X_old[C_eta, :] + X_old[C_eta, L_xi]
    x_eta_eta = X_old[R_eta, :] - 2*X_old[C_eta, :] + X_old[L_eta, :]
    x_xi_eta  = 0.25 * (X_old[R_eta, R_xi] - X_old[R_eta, L_xi] - X_old[L_eta, R_xi] + X_old[L_eta, L_xi])
    
    y_xi_xi   = Y_old[C_eta, R_xi] - 2*Y_old[C_eta, :] + Y_old[C_eta, L_xi]
    y_eta_eta = Y_old[R_eta, :] - 2*Y_old[C_eta, :] + Y_old[L_eta, :]
    y_xi_eta  = 0.25 * (Y_old[R_eta, R_xi] - Y_old[R_eta, L_xi] - Y_old[L_eta, R_xi] + Y_old[L_eta, L_xi])
    
    # Calculate Wall Foreground Controls (j=0 surface slice)
    x_xi_w  = 0.5 * (X_old[0, R_xi] - X_old[0, L_xi])
    y_xi_w  = 0.5 * (Y_old[0, R_xi] - Y_old[0, L_xi])
    gamma_w = x_xi_w**2 + y_xi_w**2
    
    x_eta_w = -y_xi_w * (delta_s / np.sqrt(gamma_w + 1e-12))
    y_eta_w =  x_xi_w * (delta_s / np.sqrt(gamma_w + 1e-12))
    
    x_xi_xi_w = X_old[0, R_xi] - 2*X_old[0, :] + X_old[0, L_xi]
    y_xi_xi_w = Y_old[0, R_xi] - 2*Y_old[0, :] + Y_old[0, L_xi]
    x_eta_eta_w = X_old[2, :] - 2*X_old[1, :] + X_old[0, :]
    y_eta_eta_w = Y_old[2, :] - 2*Y_old[1, :] + Y_old[0, :]
    
    J_w = x_xi_w * y_eta_w - y_xi_w * x_eta_w
    P_wall = (x_xi_w * x_xi_xi_w + y_xi_w * y_xi_xi_w) / (gamma_w + 1e-12)
    Q_wall = (J_w * (x_xi_w * y_eta_w - y_xi_w * x_eta_w) - gamma_w * (x_xi_w * x_eta_eta_w + y_xi_w * y_eta_eta_w)) / (gamma_w * np.sqrt(gamma_w) + 1e-12)
    
    P = np.zeros((num_eta_inner - 2, num_xi))
    Q = np.zeros((num_eta_inner - 2, num_xi))
    for row_j in range(num_eta_inner - 2):
        P[row_j, :] = P_wall * decay_factors[row_j + 1]
        Q[row_j, :] = Q_wall * decay_factors[row_j + 1]
        
    x_target = (alpha * (X_old[C_eta, R_xi] + X_old[C_eta, L_xi] + P * x_xi) + 
                gamma * (X_old[R_eta, :] + X_old[L_eta, :] + Q * x_eta) - 
                2.0 * beta * x_xi_eta) / (2.0 * (alpha + gamma))
                
    y_target = (alpha * (Y_old[C_eta, R_xi] + Y_old[C_eta, L_xi] + P * y_xi) + 
                gamma * (Y_old[R_eta, :] + Y_old[L_eta, :] + Q * y_eta) - 
                2.0 * beta * y_xi_eta) / (2.0 * (alpha + gamma))
    
    X_inner[C_eta, :] = (1.0 - omega) * X_old[C_eta, :] + omega * x_target
    Y_inner[C_eta, :] = (1.0 - omega) * Y_old[C_eta, :] + omega * y_target
    
    # Seamless closing profile at the trailing edge crease
    te_x = 0.5 * (X_inner[C_eta, 0] + X_inner[C_eta, -1])
    te_y = 0.5 * (Y_inner[C_eta, 0] + Y_inner[C_eta, -1])
    X_inner[C_eta, 0] = te_x; X_inner[C_eta, -1] = te_x
    Y_inner[C_eta, 0] = te_y; Y_inner[C_eta, -1] = te_y

# FIXED ANCHOR LABELS BELOW:
X_inner[0, :] = x_airfoil;      Y_inner[0, :] = y_airfoil
X_inner[-1, :] = x_sleeve_outer; Y_inner[-1, :] = y_sleeve_outer
X_inner[:, 0] = X_inner[:, -1];  Y_inner[:, 0] = Y_inner[:, -1]

# ==============================================================================
# 4. BLOCK 2: LINEAR CARTESIAN H-GRID TRANSITION OUTER LOOP
# ==============================================================================
x_farbox = np.zeros(num_xi)
y_farbox = np.zeros(num_xi)
for i, t in enumerate(theta):
    if np.cos(t) > 0 and np.abs(np.tan(t)) <= (y_max/x_max): 
        x_farbox[i] = x_max; y_farbox[i] = x_max * np.tan(t)
    elif np.sin(t) > 0 and np.abs(1/np.tan(t)) <= (x_max/y_max): 
        y_farbox[i] = y_max; x_farbox[i] = y_max / np.tan(t)
    elif np.cos(t) < 0 and np.abs(np.tan(t)) <= (np.abs(y_min)/np.abs(x_min)): 
        x_farbox[i] = x_min; y_farbox[i] = x_min * np.tan(t)
    else: 
        y_farbox[i] = y_min; x_farbox[i] = y_min / np.tan(t)

X_outer = np.zeros((num_eta_outer, num_xi))
Y_outer = np.zeros((num_eta_outer, num_xi))

# Expansion distribution stretching function to ease smoothly out into the box
j_out_norm = np.linspace(0, 1, num_eta_outer)
w_outer = np.sinh(2.0 * j_out_norm) / np.sinh(2.0)

for j in range(num_eta_outer):
    w = w_outer[j]
    X_outer[j, :] = (1.0 - w) * X_inner[-1, :] + w * x_farbox
    Y_outer[j, :] = (1.0 - w) * Y_inner[-1, :] + w * y_farbox

# ==============================================================================
# 5. GLOBAL MESH CONCATENATION AND PLOTTING
# ==============================================================================
# Vertically stack the two blocks along the eta axis (dropping overlapping interface row)
X_global = np.vstack([X_inner, X_outer[1:, :]])
Y_global = np.vstack([Y_inner, Y_outer[1:, :]])

plt.figure(figsize=(12, 9))
for j in range(num_eta_total):
    plt.plot(X_global[j, :], Y_global[j, :], 'b-', linewidth=0.4)
for i in range(0, num_xi, 2):  
    plt.plot(X_global[:, i], Y_global[:, i], 'r-', linewidth=0.4)

plt.axis('equal')
plt.xlim([-1.5, 4.0])  # Focus frame checking the grid transition
plt.ylim([-2.0, 2.0])
plt.title("Multi-Block O-H Hybrid Aerodynamic Grid System")
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()