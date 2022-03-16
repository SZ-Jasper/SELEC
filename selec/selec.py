# Importing Packages -----------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time

from batt_descriptor.batt_describe import *
from model.knn import *
from predictor.batt_predict import *

st.set_page_config(layout="wide")


# Logo, bc priorities --------------------------------------------
st.sidebar.image('../doc/SELEC Logo.png')


# Parameter selection --------------------------------------------
an_df = pd.DataFrame({
    'anodes': ['graphite']
    })
cath_df = pd.DataFrame({
    'cathodes': ['NMC', 'NCA', 'LFP']
    })
temp_select = pd.DataFrame({
    'temp': [15, 25, 35]
    })    
c_rate_select = pd.DataFrame({
    'c-rate': [0.5, 1.0, 2.0, 3.0]
    })
cycle_select = pd.DataFrame({
    'cycle_num': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    })

with st.sidebar.form(key = 'parameters'):
    # Anode Selector 
    an_option = st.selectbox(
        'Select an anode.',
         an_df['anodes'])

    # Cathode Selector
    cath_option = st.selectbox(
        'Select a cathode.',
         cath_df['cathodes'])

    # Temp Selector
    temp_option = st.selectbox(
        'Temperature',
         temp_select['temp'])

    # C-Rate Selector
    c_rate_option = st.selectbox(
        'C-rate',
         c_rate_select['c-rate'])

    # Cycle Selector
    cycle_option = st.selectbox(
        'Cycle Number',
         cycle_select['cycle_num'])
    
    submit_button = st.form_submit_button(label = 'Calculate')


# Output array of inputs to machine learning ----------------------
front_to_back = [an_option, cath_option, cycle_option, temp_option, c_rate_option]

'You selected:  ', an_option, ', ', cath_option,', ', str(temp_option), 'Celsius, ', \
    str(c_rate_option), 'C, and cycle number ', str(cycle_option) 


# SELEC magic -----------------------------------------------------
with st.spinner('Working SELEC magic ... '):
    report = report_gen(df_battery)
    pulled_data = backend_to_frontend(df_battery, front_to_back, report)
'...and now we\'re done!'


# Data Visualization -----------------------------------------------
# first 10 points are A123
elec_a123 = pulled_data.iloc[0:10, :]

# next 10 are Pan
elec_pan = pulled_data.iloc[10:20, :]

# last ten are LGC
elec_LGC = pulled_data.iloc[20:30, :]

col1, col2 = st.columns(2)


# Charge Capacity --------------------------------------------------

with col1:
    fig = go.Figure(data=go.Scatter3d(
        x=elec_a123['electrolyte'], 
        y=elec_a123['Cycle'], 
        z=elec_a123['Charge_Capacity (Ah)'],
        marker=dict(
            size=10,
            color = '#5271FF'
        ),
        line=dict(
            color='#5271FF',
            width=3,
            dash='dash'
        ),
        name='A123'
    ))


    fig.add_trace(
        go.Scatter3d(
            x=elec_pan['electrolyte'],
            y=elec_pan['Cycle'],
            z=elec_pan['Charge_Capacity (Ah)'],
            marker=dict(
                size=10,
                color = '#FF5757'
            ),
            line=dict(
                color='#FF5757',
                width=3,
                dash='dash'
            ),
            name='Panasonic'
        )
    )


    fig.add_trace(
        go.Scatter3d(
            x=elec_LGC['electrolyte'], 
            y=elec_LGC['Cycle'],
            z=elec_LGC['Charge_Capacity (Ah)'],
            marker=dict(
                size=10,
                color = '#FFBD59'
                #color = '#545454'
            ),
            line=dict(
                color = '#FFBD59',
                #color='#545454',
                width=3,
                dash='dash'
            ),
            name='LG Chem'
        )
    )


    fig.update_layout(title ={'text' :front_to_back[1] + ' Charge Capacity', 
                              'x' : 0.5},
                      autosize=False,
                      width=600,
                      height=600,
                      font = dict(size = 14),
                      scene = dict(
                          xaxis = dict(
                              nticks = 3,
                              title = 'Electrolyte'),
                          yaxis = dict(
                              title = 'Cycles'),
                          zaxis = dict(
                              title = 'Charge Capacity [Ah]')
                          )
                      )

    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, use_container_width=False)
    
    
with col2:
    fig1 = go.Figure(data=go.Scatter3d(
        x=elec_a123['electrolyte'], 
        y=elec_a123['Cycle'], 
        z=elec_a123['Discharge_Capacity (Ah)'],
        marker=dict(
            size=10,
            color = '#5271FF'
        ),
        line=dict(
            color='#5271FF',
            width=3,
            dash='dash'
        ),
        name='A123'
    ))

    fig1.add_trace(
        go.Scatter3d(
            x=elec_pan['electrolyte'],
            y=elec_pan['Cycle'],
            z=elec_pan['Discharge_Capacity (Ah)'],
            marker=dict(
                size=10,
                color = '#FF5757'
            ),
            line=dict(
                color='#FF5757',
                width=3,
                dash='dash'
            ),
            name='Panasonic'
            )
        )

    fig1.add_trace(
        go.Scatter3d(
            x=elec_LGC['electrolyte'], 
            y=elec_LGC['Cycle'],
            z=elec_LGC['Discharge_Capacity (Ah)'],
            marker=dict(
                size=10,
                color = '#FFBD59'
                #color = '#545454'
            ),
            line=dict(
                color = '#FFBD59',
                #color='#545454',
                width=3,
                dash='dash'
            ),
            name='LG Chem'
            )
        )

    fig1.update_layout(title ={'text' :front_to_back[1] + ' Discharge Capacity', 
                              'x' : 0.5},
                      autosize=False,
                      width=600,
                      height=600,
                      font = dict(size = 14),
                      scene = dict(
                          xaxis = dict(
                              nticks = 3,
                              title = 'Electrolyte'),
                          yaxis = dict(
                              title = 'Cycles'),
                          zaxis = dict(
                              title = 'Discharge Capacity [Ah]')
                          )
                      )

    fig1.update_yaxes(automargin=True)
    st.plotly_chart(fig1,use_container_width=False)

# Just adding white space
st.empty()


# Coulombic Efficiency
fig2 = go.Figure(data=go.Scatter(x=elec_a123['Cycle'], 
                                 y=elec_a123['Coulombic_Efficiency (%)'],
                                 marker=dict(
                                 size=10,
                                 color = '#5271FF'
                                ),
                                line=dict(
                                    color='#5271FF',
                                    width=3,
                                    dash='dash'
                                ),
                                name='A123')
                )

fig2.add_trace(go.Scatter(x=elec_pan['Cycle'], 
                                 y=elec_pan['Coulombic_Efficiency (%)'],
                                 marker=dict(
                                     size=15,
                                     color = '#FF5757',
                                     symbol = 'triangle-up'
                                 ),
                                 line=dict(
                                     color='#FF5757',
                                     width=3,
                                     dash='dash'
                                 ),
                                 name='Panasonic'
                         )
              )

fig2.add_trace(go.Scatter(x=elec_LGC['Cycle'], 
                          y=elec_LGC['Coulombic_Efficiency (%)'],
                          marker=dict(
                              size=10,
                              color = '#FFBD59',
                              #color = '#545454',
                              symbol = 'square'
                          ),
                          line=dict(
                              color = '#FFBD59',
                              #color='#545454',
                              width=3,
                              dash='dash'
                          ),
                          name='LG Chem'
                         )
              )

fig2.update_layout(title ={'text' :front_to_back[1] + ' Coulombic Efficiency', 
                          'x' : 0.45},
                   font = dict(size = 15),
                   xaxis = dict(
                       title = 'Number of Cycles'),
                   yaxis = dict(
                       title = 'CE (%)')
                  )
st.plotly_chart(fig2)

# Energy Efficiency
fig3 = go.Figure(data=go.Scatter(x=elec_a123['Cycle'], 
                                 y=elec_a123['Energy_Efficiency (%)'],
                                 marker=dict(
                                 size=10,
                                 color = '#5271FF'
                                ),
                                line=dict(
                                    color='#5271FF',
                                    width=3,
                                    dash='dash'
                                ),
                                name='A123')
                )

fig3.add_trace(go.Scatter(x=elec_pan['Cycle'], 
                                 y=elec_pan['Energy_Efficiency (%)'],
                                 marker=dict(
                                     size=15,
                                     color = '#FF5757',
                                     symbol = 'triangle-up'
                                 ),
                                 line=dict(
                                     color='#FF5757',
                                     width=3,
                                     dash='dash'
                                 ),
                                 name='Panasonic'
                         )
              )

fig3.add_trace(go.Scatter(x=elec_LGC['Cycle'], 
                          y=elec_LGC['Energy_Efficiency (%)'],
                          marker=dict(
                              size=10,
                              color = '#FFBD59',
                              #color = '#545454',
                              symbol = 'square'
                          ),
                          line=dict(
                              color = '#FFBD59',
                              #color='#545454',
                              width=3,
                              dash='dash'
                          ),
                          name='LG Chem'
                         )
              )

fig3.update_layout(title ={'text' :front_to_back[1] + ' Energy Efficiency', 
                          'x' : 0.45},
                   font = dict(size = 15),
                   xaxis = dict(
                       title = 'Number of Cycles'),
                   yaxis = dict(
                       title = 'EE (%)')
                  )
st.plotly_chart(fig3)


# more white space
st.empty()


col3, col4 = st.columns(2)
# Charge Energy --------------------------------------------------

with col3:
    fig4 = go.Figure(data=go.Scatter3d(
        x=elec_a123['electrolyte'], 
        y=elec_a123['Cycle'], 
        z=elec_a123['Charge_Energy (Wh)'],
        marker=dict(
            size=10,
            color = '#5271FF'
        ),
        line=dict(
            color='#5271FF',
            width=3,
            dash='dash'
        ),
        name='A123'
    ))


    fig4.add_trace(
        go.Scatter3d(
            x=elec_pan['electrolyte'],
            y=elec_pan['Cycle'],
            z=elec_pan['Charge_Energy (Wh)'],
            marker=dict(
                size=10,
                color = '#FF5757'
            ),
            line=dict(
                color='#FF5757',
                width=3,
                dash='dash'
            ),
            name='Panasonic'
        )
    )


    fig4.add_trace(
        go.Scatter3d(
            x=elec_LGC['electrolyte'], 
            y=elec_LGC['Cycle'],
            z=elec_LGC['Charge_Energy (Wh)'],
            marker=dict(
                size=10,
                color = '#FFBD59'
            ),
            line=dict(
                color = '#FFBD59',
                width=3,
                dash='dash'
            ),
            name='LG Chem'
        )
    )


    fig4.update_layout(title ={'text' :front_to_back[1] + ' Charge Energy', 
                              'x' : 0.5},
                      autosize=False,
                      width=600,
                      height=600,
                      font = dict(size = 14),
                      scene = dict(
                          xaxis = dict(
                              nticks = 3,
                              title = 'Electrolyte'),
                          yaxis = dict(
                              title = 'Cycles'),
                          zaxis = dict(
                              title = 'Charge Energy [Wh]')
                          )
                      )

    fig4.update_yaxes(automargin=True)
    st.plotly_chart(fig4, use_container_width=False)
    
    
with col4:
    fig5 = go.Figure(data=go.Scatter3d(
        x=elec_a123['electrolyte'], 
        y=elec_a123['Cycle'], 
        z=elec_a123['Discharge_Energy (Wh)'],
        marker=dict(
            size=10,
            color = '#5271FF'
        ),
        line=dict(
            color='#5271FF',
            width=3,
            dash='dash'
        ),
        name='A123'
    ))

    fig5.add_trace(
        go.Scatter3d(
            x=elec_pan['electrolyte'],
            y=elec_pan['Cycle'],
            z=elec_pan['Discharge_Energy (Wh)'],
            marker=dict(
                size=10,
                color = '#FF5757'
            ),
            line=dict(
                color='#FF5757',
                width=3,
                dash='dash'
            ),
            name='Panasonic'
            )
        )

    fig5.add_trace(
        go.Scatter3d(
            x=elec_LGC['electrolyte'], 
            y=elec_LGC['Cycle'],
            z=elec_LGC['Discharge_Energy (Wh)'],
            marker=dict(
                size=10,
                color = '#FFBD59'
            ),
            line=dict(
                color = '#FFBD59',
                width=3,
                dash='dash'
            ),
            name='LG Chem'
            )
        )

    fig5.update_layout(title ={'text' :front_to_back[1] + ' Discharge Energy', 
                              'x' : 0.5},
                      autosize=False,
                      width=600,
                      height=600,
                      font = dict(size = 14),
                      scene = dict(
                          xaxis = dict(
                              nticks = 3,
                              title = 'Electrolyte'),
                          yaxis = dict(
                              title = 'Cycles'),
                          zaxis = dict(
                              title = 'Discharge Energy [Wh]')
                          )
                      )

    fig5.update_yaxes(automargin=True)
    st.plotly_chart(fig5,use_container_width=False)