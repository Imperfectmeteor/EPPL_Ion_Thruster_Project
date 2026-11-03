import sys
import streamlit.web.cli
import streamlit.runtime.runtime

# Streamlit system command run for the web Client
# http://localhost:8501
if not streamlit.runtime.runtime.Runtime.exists():
    sys.argv[:] = ['streamlit', 'run', 'Testing 2.py']
    sys.exit(streamlit.web.cli.main())

# Math library Imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import streamlit as st

# 4-Digit Airfoil Geometry Generator
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
        # 1. Force the input into a string and strip spaces
        clean_code = str(self.code).strip()
        
        # 2. If it has a decimal point from an accidental float input (e.g., "2412.0"), strip it
        if '.' in clean_code:
            clean_code = clean_code.split('.')[0]
            
        # 3. Stop execution gracefully if the code isn't 4 digits yet
        if len(clean_code) != 4 or not clean_code.isdigit():
            st.warning("Please enter a valid 4-digit NACA code (e.g., 2412).")
            st.stop()
            
        # 4. Safely parse out individual digits
        NACA_n = [float(x) for x in clean_code]
        
        # 5. Extract structural attributes using explicit indices
        self.m = (NACA_n[0] * 0.01)  # Maximum Camber
        self.p = (NACA_n[1] * 0.1)   # Position of Maximum Camber
        self.T = (((NACA_n[2] * 10) + NACA_n[3]) * 0.01)  # Maximum Thickness
    
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
        Theta_d = np.arctan(dy_c)  # Derivative Angle
        
        # Thickness distribution
        t_n = self.t / self.c  # Normalizing t w/r c
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
        
#-----------------------------------------------------------------------------------------
# Airfoil Plotting
def Airfoilplot(Airfoil_Geo):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(Airfoil_Geo.X_u, Airfoil_Geo.Y_u, color='red', linewidth=2, label='Upper Surface')
    ax.plot(Airfoil_Geo.X_l, Airfoil_Geo.Y_l, color='red', linewidth=2, label='Lower Surface')
    ax.plot(Airfoil_Geo.t, Airfoil_Geo.y_c, linestyle='--', color='blue', alpha=0.7, label='Camber Line')
    ax.set_title(f'NACA {Airfoil_Geo.code} Airfoil Geometry Generation', fontsize=14)
    ax.set_xlabel('Chord Position (x/c)', fontsize=12)
    ax.set_ylabel('Thickness / Camber (y/c)', fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    st.pyplot(fig)

# Streamlit UI Header
st.title("NACA 4-Digit Airfoil Generator")
st.markdown("Enter parameters below to dynamically update the airfoil profile geometry.")

# Airfoil Parameter Inputs (Forced text_input for code string filtering)
code_input = st.text_input('Enter a 4-digit Airfoil Code: ', value="2412")
chord_input = st.number_input('Enter a Desired Chord Length: ', min_value=0.1, value=1.0, step=0.1)
precision_input = st.number_input('Enter the Desired Geometric Precision: ', min_value=10, value=100, step=10)

# Airfoil Class Call & Outputs
Airfoil_Geo = NACA4DigitAirfoil(code_input, chord_input, precision_input)
Airfoilplot(Airfoil_Geo)