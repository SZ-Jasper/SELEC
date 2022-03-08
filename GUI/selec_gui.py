# Importing Packages -----------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np

import time


# Logo, bc priorities --------------------------------------------
# this will eventuallly tie into the github
st.sidebar.image('SELEC Logo.png')


# Cathode Selector -----------------------------------------------
cath_select = pd.DataFrame({
    'cathodes': ['NMC', 'NCA', 'LFP']
    })

cath_option = st.sidebar.selectbox(
    'Select a cathode.',
     cath_select['cathodes'])


# Temp Selector ------------------------------------------------
# st.sidebar.text_input('Temperature: ')
# this would be for typed input rather than drop-down

temp_select = pd.DataFrame({
    'temp': [15, 25, 35]
    })

temp_option = st.sidebar.selectbox(
    'Temperature',
     temp_select['temp'])


# C-Rate Selector ------------------------------------------------
# st.sidebar.text_input('C-Rate: ')
c_rate_select = pd.DataFrame({
    'c-rate': [0.5, 1.0, 2.0, 3.0]
    })

c_rate_option = st.sidebar.selectbox(
    'C-rate',
     c_rate_select['c-rate'])

# Cycle Selector ------------------------------------------------
# st.sidebar.text_input('Cycle Number: ')
cycle_select = pd.DataFrame({
    'cycle_num': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    })

cycle_option = st.sidebar.selectbox(
    'Cycle Number',
     cycle_select['cycle_num'])


# Start Calculation ----------------------------------------------
st.sidebar.button('Calculate')


# Output array of inputs to machine learning ----------------------
front_to_back = [cath_option, temp_option, c_rate_option, cycle_option]

'You selected: ', cath_option,', ', str(temp_option), 'Celsius, ', \
    str(c_rate_option), 'C, and cycle number ', str(cycle_option) 


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