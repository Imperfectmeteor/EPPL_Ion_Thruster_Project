import numpy as np
#from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#Cylindrical Streamline Plotter
class Cylinderstream:
    def __init__(self):
        #Mesh Grid
        self.x = np.linspace(-4, 4, 200)
        self.y = np.linspace(-3, 3, 200)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        #Initial Conditions
        self.U_i = 1 #Streamflow
        self.R = 1   #Radius
        self.p = 0   #Plotting Method
        
        #Flow Equations
        self.U = self.U_i * (1-(self.R**2 * ((self.X**2 - self.Y**2)/((self.X**2 + self.Y**2)**2))))
        self.V = -2*self.U_i*((self.R**2)*(self.X*self.Y/((self.X**2+self.Y**2)**2)))
        
        #Dynamic Arrow Size & Color, and Velocity Function Filtering
        self.speed = np.sqrt((self.U**2) + (self.V**2))
        self.Internal = ((self.X**2) + (self.Y**2)) < (self.R**2)
        self.U_m = np.ma.masked_where(self.Internal, self.U)
        self.V_m = np.ma.masked_where(self.Internal, self.V)
        self.speed_m = np.ma.masked_where(self.Internal, self.speed)
        
        #Streamline and Body Plotting
        plotmethod(self)
    
    #Data Plotting method, p=0:colorgradient, p=1:streamlines
    def plotmethod(self):
        fig, bx = plt.subplots(figsize=(12, 8))
        self.norm = mcolors.Normalize(vmin = 0, vmax = self.U_i)
        if self.p == 1:
            cf = bx.contourf(self.X, self.Y,
                         speed_m,
                         levels=1000,
                         cmap='viridis',
                         norm = self.norm)
            fig.colorbar(cf, bx=self.bx, label='Velocity')
        if self.p == 2:
            strm = bx.streamplot(self.X, self.Y, self.U_m, self.V_m, 
                        color=self.speed_m,          # Color mapped to local speed
                        cmap='viridis',       # Vivid colormap
                        norm = self.norm, #Normalizes the color gradient with the speed of the flow
                        linewidth=1 + 0.5 * (self.speed_m / self.speed_m.max()), # Dynamic thickness of lines
                        density=5)          # Controls line closeness (default=1)
            fig.colorbar(strm.lines, bx=bx, label='Velocity')
        
        #Cylinder Plotting
        self.T = np.linspace(0, 2*np.pi, 100)
        self.xt = self.R * np.cos(T)
        self.yt = self.R * np.sin(T)
        self.circ = bx.plot(xt, yt)
        
        #Final Plot Settings
        bx.set_title('Flow Over A Cylinder') 
        bx.set_aspect('equal', adjustable='datalim')
        bx.grid(True)
        plt.show()
#-----------------------------------------------------------------------------------------

#4-Digit Airfoil Geometry Generator
class NACA4DigitAirfoil:
    
    #Runs the class's functions and attaches inputs
    def __init__(self, code: str, chord: float, precision: int):
        self.code = code
        self.c = chord
        self.precision = precision
        self.parameters()
        self.t = np.linspace(0, self.c, self.precision, dtype=float)
        self.compute_geometry()
    
    #Defines the airfoil parameters from the NACA code
    def parameters(self):
        #NACA Airfoil Parameters
        NACA_n = [float(x) for x in self.code]
        
        self.m = (NACA_n[0] * 0.01) #Maximum Camber
        self.p = (NACA_n[1] * 0.1) #Position of Maximum Camber
        self.T = (((NACA_n[2] * 10) + NACA_n[3]) * 0.01) #Maximum Thickness
    
    #Computes the mean camber line for the given airfoil
    def camber_line(self, t):
        if self.m == 0 or self.p == 0:
            return np.zeros_like(t, dtype=float)
        
        t_norm = t / self.c
        con_list = [(t >= 0) & (t <= self.p * self.c),(t >= self.p * self.c) & (t <= self.c)]
        func_list = [(self.m/self.p**2)*((2*self.p*t_norm)-(t_norm**2)),
                     (self.m/((1- self.p)**2))*((1-(2*self.p))+(2*self.p*t_norm)-(t_norm**2))]
        return np.select(con_list, func_list, 0.0)
    
    #Computes the "physical" geometry of the airfoil
    def compute_geometry(self):
        self.y_c = self.camber_line(self.t)
        
        dy_c = np.gradient(self.y_c, self.t)
        #y_diff = sci.differentiate.derivative(lambda x: y_c(x, m, p, c), t)
        #dy_c = y_diff.df
        Theta_d = np.arctan(dy_c) #Derivative Angle
        
        #Thickness ditribution
        t_n = self.t / self.c #Normalizing t w/r c
        y_t = 5*(self.T * self.c)*((0.2969*np.sqrt(t_n))-
                   (0.1260*t_n)-
                   (0.3516*(t_n**2))+
                   (0.2843*(t_n**3))-
                   (0.1015*(t_n**4)))
        
        #Upper and Lower Airfoil Geometric Functions
        self.X_u = (self.t - (y_t * np.sin(Theta_d)))
        self.Y_u = self.y_c + (y_t * np.cos(Theta_d))
        
        self.X_l = (self.t + (y_t * np.sin(Theta_d)))
        self.Y_l = self.y_c - (y_t * np.cos(Theta_d))
        
#-----------------------------------------------------------------------------------------
#Airfoil Plotting
def Airfoilplot(Airfoil_Geo):
    fig, ax = plt.subplots(figsize=(12, 8))
    Af_upper = ax.plot(Airfoil_Geo.X_u, Airfoil_Geo.Y_u, color='red')
    Af_lower = ax.plot(Airfoil_Geo.X_l, Airfoil_Geo.Y_l, color='red')
    Chord_line = ax.plot(Airfoil_Geo.t, Airfoil_Geo.y_c, linestyle='--')
    ax.set_title('Flow Over An Airfoil')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True)
    plt.show()

#Airfoil Parameter Inputs
code_input = input('Enter a 4-digit Airfoil Code: ')
chord_input = float(input('Enter a Desired Chord Length: '))
precision_input = int(input('Enter the Desired Geometric Precision: '))

#Airfoil Class Call & Outputs
Airfoil_Geo = NACA4DigitAirfoil(code_input, chord_input, precision_input)
X_u, Y_u, X_l, Y_l = Airfoil_Geo.X_u, Airfoil_Geo.Y_u, Airfoil_Geo.X_l, Airfoil_Geo.Y_l
Airfoilplot(Airfoil_Geo)