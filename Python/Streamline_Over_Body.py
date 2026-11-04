import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#Cylindrical Streamline Plotter
class Cylinderstream:
    def __init__(self, p):
        #Mesh Grid
        self.x = np.linspace(-4, 4, 200)
        self.y = np.linspace(-3, 3, 200)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        #Initial Conditions
        self.U_i = 1 #Streamflow
        self.R = 1   #Radius
        self.p = p   #Plotting Method
        self.T = np.linspace(0, 2*np.pi, 200)
        
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
        self.plotmethod()
    
    #Data Plotting method, p=0:colorgradient, p=1:streamlines
    def plotmethod(self):
        fig, bx = plt.subplots(figsize=(12, 8))
        self.norm = mcolors.Normalize(vmin = 0, vmax = self.U_i)
        if self.p == 1:
            cf = bx.contourf(self.X, self.Y,
                         self.speed_m,
                         levels=1000,
                         cmap='viridis',
                         norm = self.norm)
            fig.colorbar(cf, ax=bx, label='Velocity')
        if self.p == 2:
            strm = bx.streamplot(self.X, self.Y, self.U_m, self.V_m, 
                        color=self.speed_m,          # Color mapped to local speed
                        cmap='viridis',       # Vivid colormap
                        norm = self.norm, #Normalizes the color gradient with the speed of the flow
                        linewidth=1 + 0.5 * (self.speed_m / self.speed_m.max()), # Dynamic thickness of lines
                        density=5)          # Controls line closeness (default=1)
            fig.colorbar(strm.lines, ax=bx, label='Velocity')
        
        #Cylinder Plotting
        self.T = np.linspace(0, 2*np.pi, 100)
        self.xt = self.R * np.cos(self.T)
        self.yt = self.R * np.sin(self.T)
        bx.plot(self.xt, self.yt, color='black', linewidth=2, zorder=5)
        
        #Final Plot Settings
        bx.set_title('Flow Over A Cylinder') 
        bx.set_aspect('equal', adjustable='datalim')
        bx.grid(True)
        plt.show()
#-----------------------------------------------------------------------------------------
Cylinderstream(2)