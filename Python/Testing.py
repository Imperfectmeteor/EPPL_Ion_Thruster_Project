import sys
import streamlit.web.cli
import streamlit.runtime.runtime

#Streamlit system command run for the web Client
#http://localhost:8501
if not streamlit.runtime.runtime.Runtime.exists():
    sys.argv[:] = ['streamlit', 'run', 'Testing.py']
    sys.exit(streamlit.web.cli.main())

#Math library Imports
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import streamlit as st

#-----Equations-----



fig, sc = plt.subplots()
sc.quiver(, , dx, dy)

st.pyplot(fig)