# Importing Packages -----------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go
import plotly.express as px

import time


# Logo, bc priorities --------------------------------------------
# this will eventuallly tie into the github
st.sidebar.image('SELEC Logo.png')


# Currently, this code wil execute as soon as you select something
# I'd like to change it to where it only executes when you press "calculate"

# Anode Selector -----------------------------------------------
an_df = pd.DataFrame({
    'anodes': ['graphite']
    })

an_option = st.sidebar.selectbox(
    'Select an anode.',
     an_df['anodes'])

# Cathode Selector -----------------------------------------------
cath_df = pd.DataFrame({
    'cathodes': ['NMC', 'NCA', 'LFP']
    })

cath_option = st.sidebar.selectbox(
    'Select a cathode.',
     cath_df['cathodes'])

# Temp Selector ------------------------------------------------

temp_select = pd.DataFrame({
    'temp': [15, 25, 35]
    })

temp_option = st.sidebar.selectbox(
    'Temperature',
     temp_select['temp'])


# C-Rate Selector ------------------------------------------------
c_rate_select = pd.DataFrame({
    'c-rate': [0.5, 1.0, 2.0, 3.0]
    })

c_rate_option = st.sidebar.selectbox(
    'C-rate',
     c_rate_select['c-rate'])

# Cycle Selector ------------------------------------------------
cycle_select = pd.DataFrame({
    'cycle_num': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    })

cycle_option = st.sidebar.selectbox(
    'Cycle Number',
     cycle_select['cycle_num'])


# Start Calculation ----------------------------------------------
st.sidebar.button('Calculate')


# Output array of inputs to machine learning ----------------------
front_to_back = [an_option, cath_option, cycle_option, temp_option, c_rate_option]

'You selected: ', an_option, ', ', cath_option,', ', str(temp_option), 'Celsius, ', \
    str(c_rate_option), 'C, and cycle number ', str(cycle_option) 


# Progress Bar ----------------------------------------------------
'Starting a long computation...'
latest_iteration = st.empty()
bar = st.progress(0)

# we have to update this to reflect the actual computational time
n = 100
for i in range(n):
  # Update the progress bar with each iteration.
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'


# Data Visualization -----------------------------------------------
sns.set_context('talk')

x = [1,1,1] # input, strings (ex: 'NMC')
y = [50,100,150] # input
z = [13, 69, 21] # output

fig = px.scatter_3d(x=x, y=y, z=z,
                    color=x, 
                    labels = {'x': 'Battery System', 'y': 'Cycle Number',
                              'z': 'Output'})

fig.update_layout(title ={'text' :'help me', 
                          'x' : 0.5},
                 scene = dict(
                     xaxis = dict(
                         nticks = 3,
                         ticktext = ['NMC', 'NFP', 'LCA'],
                     tickvals = [0, 1, 2]))
                 )

st.plotly_chart(fig)


# data frames --> output, MSE
# we're interested in a 3D plot
# axes: battery system, cycle, output, 