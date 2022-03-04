# Importing Packages -----------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np

import time


# Logo, bc priorities --------------------------------------------
# this will eventuallly tie into the github
st.sidebar.image('SELEC Logo.png')


# Cathode Selector -----------------------------------------------
df = pd.DataFrame({
    'cathodes': ['NMC', 'NCA', 'LFP'], 
    })

option = st.sidebar.selectbox(
    'Select a cathode.',
     df['cathodes'])

'You selected: ', option

# cycle 50 to 500 increments of 50 
# temp 15 25 35
# c rate 0.5 1 2 3 

#     'temp': [15, 25, 35],
#     'c-rate': [0.5, 1.0, 2.0, 3.0],
#     'cycle_num': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],

# Inputting data ------------------------------------------------
st.sidebar.text_input('Temperature: ')
st.sidebar.text_input('C-Rate: ')
st.sidebar.text_input('Cycle Number: ')


# Start Calculation ----------------------------------------------
st.sidebar.button('Calculate')


# Progress Bar ----------------------------------------------------
'Starting a long computation...'
# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

n = 100
for i in range(n):
  # Update the progress bar with each iteration.
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'


# Data Visualization -----------------------------------------------

'Open Circuit Voltage'
OCV = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(OCV)