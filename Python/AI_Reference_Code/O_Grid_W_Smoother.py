import numpy as np
import matplotlib.pyplot as plt
#Plotly
#
#GNUPlot

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

# =====================================================================
# GRID GENERATION INTERFACE
# =====================================================================

# 1. Instantiate the Airfoil Geometry (e.g., Cambered NACA 2412)
chord_length = 1.0
precision_points = 50  # Results in 2 * precision_points - 1 wrap-around points
airfoil = NACA4DigitAirfoil(code="2412", chord=chord_length, precision=precision_points)

# 2. Combine into a continuous wrap-around loop (Trailing Edge -> Upper -> Leading Edge -> Lower -> Trailing Edge)
# We flip the upper surface array to trace from Trailing Edge (x=c) to Leading Edge (x=0)
x_airfoil = np.concatenate([np.flip(airfoil.X_u), airfoil.X_l[1:]])
y_airfoil = np.concatenate([np.flip(airfoil.Y_u), airfoil.Y_l[1:]])

num_xi = len(x_airfoil)  # Number of grid lines wrapped around the airfoil
num_eta = 50             # Number of grid lines moving outward to the farfield / resolution

# 3. Create a circular Outer Boundary (Far-field)
x_far = np.zeros(num_xi)
y_far = np.zeros(num_xi)

x_min, x_max = -30.0 * chord_length, 30.0 * chord_length
y_min, y_max = -30.0 * chord_length, 30.0 * chord_length

r_farfield = 5.0 * chord_length
theta = np.linspace(0, 2 * np.pi, num_xi)

for i, t in enumerate(theta):
    # Determine which quadrant/side of the box the ray points toward
    if np.cos(t) > 0 and np.abs(np.tan(t)) <= (y_max/x_max): # Right face
        x_far[i] = x_max
        y_far[i] = x_max * np.tan(t)
    elif np.sin(t) > 0 and np.abs(1/np.tan(t)) <= (x_max/y_max): # Top face
        y_far[i] = y_max
        x_far[i] = y_max / np.tan(t)
    elif np.cos(t) < 0 and np.abs(np.tan(t)) <= (np.abs(y_min)/np.abs(x_min)): # Left face
        x_far[i] = x_min
        y_far[i] = x_min * np.tan(t)
    else: # Bottom face
        y_far[i] = y_min
        x_far[i] = y_min / np.tan(t)

# 4. Initialize Coordinates Matrices (Algebraic Mesh / TFI)
X = np.zeros((num_eta, num_xi))
Y = np.zeros((num_eta, num_xi))

# Calculate a stretching factor (b) to compress the distribution near j=0
# Using an exponential blending weight instead of linear
j_norm = np.linspace(0, 1, num_eta)
stretch_param = 5  # Higher values pack more cells tighter to the airfoil

# Generate hyperbolic-like distribution weights
w_stretched = np.sinh(stretch_param * j_norm) / np.sinh(stretch_param)

for j in range(num_eta):
    # Linear blending weight moving from 0 (airfoil surface) to 1 (farfield boundary)
    w = w_stretched[j]
    X[j, :] = (1.0 - w) * x_airfoil + w * x_far
    Y[j, :] = (1.0 - w) * y_airfoil + w * y_far

# 5. Elliptic Grid Smoother Loop (Solving Laplace Equation in Curvilinear Coordinates)
iterations = 500
omega = 1.0  # Relaxation factor (Gauss-Seidel = 1.0, SOR > 1.0)

#Indexing configs for 2D Mesh
C = slice(1, -1)
L = slice(0, -2)
R = slice(2, None)

for it in range(iterations):
    # Enforce periodic boundary matching on outer seams before taking a matrix copy
    X[:, 0] = X[:, -1]
    Y[:, 0] = Y[:, -1]
    
    # Create copies for interior coordinate evaluation
    X_old = X.copy()
    Y_old = Y.copy()
    
    # First order grid metrics (Central differences)
    x_xi  = 0.5 * (X_old[C, R] - X_old[C, L])
    y_xi  = 0.5 * (Y_old[C, R] - Y_old[C, L])
    x_eta = 0.5 * (X_old[R, C] - X_old[L, C])
    y_eta = 0.5 * (Y_old[R, C] - Y_old[L, C])
    
    # Transformation coefficients
    alpha = x_eta**2 + y_eta**2
    beta  = x_xi * x_eta + y_xi * y_eta
    gamma = x_xi**2 + y_xi**2
    
    # Second order derivatives
    x_xi_xi   = X_old[C, R] - 2*X_old[C, C] + X_old[C, L]
    x_eta_eta = X_old[R, C] - 2*X_old[C, C] + X_old[L, C]
    x_xi_eta  = 0.25 * (X_old[R, R] - X_old[R, L] - X_old[L, R] + X_old[L, L])
    
    y_xi_xi   = Y_old[C, R] - 2*Y_old[C, C] + Y_old[C, L]
    y_eta_eta = Y_old[R, C] - 2*Y_old[C, C] + Y_old[L, C]
    y_xi_eta  = 0.25 * (Y_old[R, R] - Y_old[R, L] - Y_old[L, R] + Y_old[L, L])
    
    #Thomas-Middlecoff Grid Source Terms (P and Q)
    #P = (x_xi * x_xi_xi + y_xi * y_xi_xi) / (x_xi**2 + y_xi**2 + 1e-12)
    #Q = (x_eta * x_eta_eta + y_eta * y_eta_eta) / (x_eta**2 + y_eta**2 + 1e-12)
    
    # Elliptic PDE formulation solutions
    x_target = (alpha * (X_old[C, R] + X_old[C, L]) + 
                gamma * (X_old[R, C] + X_old[L, C]) - 
                2.0 * beta * x_xi_eta) / (2.0 * (alpha + gamma))
                
    y_target = (alpha * (Y_old[C, R] + Y_old[C, L]) + 
                gamma * (Y_old[R, C] + Y_old[L, C]) - 
                2.0 * beta * y_xi_eta) / (2.0 * (alpha + gamma))
    # Update coordinate matrices using SOR
    X[C, C] = (1.0 - omega) * X_old[C, C] + omega * x_target
    Y[C, C] = (1.0 - omega) * Y_old[C, C] + omega * y_target
    
    # Explicit periodic boundary condition handling for closed O-grids
    X[:, 0] = X[:, -1]
    Y[:, 0] = Y[:, -1]

# 6. Visualization
plt.figure(figsize=(10, 8))
# Plot constant eta lines (wraps around airfoil)
for j in range(num_eta):
    plt.plot(X[j, :], Y[j, :], 'b-', linewidth=0.4)
# Plot constant xi lines (radiating outwards)
for i in range(0, num_xi, 2):  # stride by 2 for plot clarity
    plt.plot(X[:, i], Y[:, i], 'r-', linewidth=0.4)

plt.axis('equal')
plt.xlim([-1.0, 3.0])
plt.ylim([-2.0, 2.0])
plt.title(f'Elliptic O-Grid Generation for NACA {airfoil.code}')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()