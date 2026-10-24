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

# =====================================================================
# GRID GENERATION INTERFACE
# =====================================================================

# 1. Instantiate the Airfoil Geometry (e.g., Cambered NACA 2412)
chord_length = 1.0
precision_points = 20  # Results in 2 * precision_points - 1 wrap-around points
airfoil = NACA4DigitAirfoil(code="2412", chord=chord_length, precision=precision_points)

# 2. Combine into a continuous wrap-around loop (Trailing Edge -> Upper -> Leading Edge -> Lower -> Trailing Edge)
# We flip the upper surface array to trace from Trailing Edge (x=c) to Leading Edge (x=0)
x_airfoil = np.concatenate([np.flip(airfoil.X_u), airfoil.X_l[1:]])
y_airfoil = np.concatenate([np.flip(airfoil.Y_u), airfoil.Y_l[1:]])

num_xi = len(x_airfoil)  # Number of grid lines wrapped around the airfoil
num_eta = 50             # Number of grid lines moving outward to the farfield / resolution

# 3. Create a circular Outer Boundary (Far-field)
r_farfield = 5.0 * chord_length
# Match angles with wrap points around the airfoil mapping context
theta = np.linspace(0, 2 * np.pi, num_xi)
x_far = chord_length / 2.0 + r_farfield * np.cos(theta)
y_far = r_farfield * np.sin(theta)

# 4. Initialize Coordinates Matrices (Algebraic Mesh / TFI)
X = np.zeros((num_eta, num_xi))
Y = np.zeros((num_eta, num_xi))

for j in range(num_eta):
    # Linear blending weight moving from 0 (airfoil surface) to 1 (farfield boundary)
    w = j / (num_eta - 1)
    X[j, :] = (1.0 - w) * x_airfoil + w * x_far
    Y[j, :] = (1.0 - w) * y_airfoil + w * y_far

# 5. Elliptic Grid Smoother Loop (Solving Laplace Equation in Curvilinear Coordinates)
iterations = 500
omega = 1.0  # Relaxation factor (Gauss-Seidel = 1.0, SOR > 1.0)

for it in range(iterations):
    # Create copies for interior coordinate evaluation
    X_old = X.copy()
    Y_old = Y.copy()
    
    for j in range(1, num_eta - 1):
        for i in range(1, num_xi - 1):
            # First order grid metrics (Central differences)
            x_xi  = 0.5 * (X[j, i+1] - X[j, i-1])
            y_xi  = 0.5 * (Y[j, i+1] - Y[j, i-1])
            x_eta = 0.5 * (X[j+1, i] - X[j-1, i])
            y_eta = 0.5 * (Y[j+1, i] - Y[j-1, i])
            
            # Transformation coefficients (Alpha, Beta, Gamma)
            alpha = x_eta**2 + y_eta**2
            beta  = x_xi * x_eta + y_xi * y_eta
            gamma = x_xi**2 + y_xi**2
            
            # Second order derivatives
            x_xi_xi   = X[j, i+1] - 2*X[j, i] + X[j, i-1]
            x_eta_eta = X[j+1, i] - 2*X[j, i] + X[j-1, i]
            x_xi_eta  = 0.25 * (X[j+1, i+1] - X[j+1, i-1] - X[j-1, i+1] + X[j-1, i-1])
            
            y_xi_xi   = Y[j, i+1] - 2*Y[j, i] + Y[j, i-1]
            y_eta_eta = Y[j+1, i] - 2*Y[j, i] + Y[j-1, i]
            y_xi_eta  = 0.25 * (Y[j+1, i+1] - Y[j+1, i-1] - Y[j-1, i+1] + Y[j-1, i-1])
            
            # Elliptic PDE formulation solutions
            x_target = (alpha * (X[j, i+1] + X[j, i-1]) + gamma * (X[j+1, i] + X[j-1, i]) - 2.0 * beta * x_xi_eta) / (2.0 * (alpha + gamma))
            y_target = (alpha * (Y[j, i+1] + Y[j, i-1]) + gamma * (Y[j+1, i] + Y[j-1, i]) - 2.0 * beta * y_xi_eta) / (2.0 * (alpha + gamma))
            
            # Update coordinate matrices using SOR
            X[j, i] = (1.0 - omega) * X[j, i] + omega * x_target
            Y[j, i] = (1.0 - omega) * Y[j, i] + omega * y_target
            
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