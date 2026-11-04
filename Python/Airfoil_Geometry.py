import numpy as np
#from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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