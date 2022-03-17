# SELEC

<img src=https://github.com/SZ-Jasper/SELEC/blob/main/doc/SELEC%20Logo.png width=300 p align="right">

The SELEC project (Maching Learning for Selecting Electrolytes) aims to produce a machine learning program that can return the optimized electrolyte for a given battery system. Considering the continuously growing research interest in the battery industry, tons of battery experiments are being conducted, and thus numerous battery datasets are available for machine learning study. In this project, the program is built up with a regression model (categorize & predict data) and a graphical user interface (for electrode determination & output display). The model will be trained with a given battery dataset that involves information of electrodes, electrolytes, battery performances in different aspects. The trained model will then be validated and tested with spliited dataset and the final prediction will be stored in a local file. The users are thus able to access the GUI by specifying battery parameters of interest and the program will returen the best combination based on the prediction result from the previously trained dataset. 


-----
## Software Dependencies
The SELEC predictor is available on the web and python is not required. Please see *Interface Instructions* for more information. 
For those who would like to run the jupyter and python files, please ensure you have the following:
- Python 3.7
- Python packages listed in `environment.yml`and Installation section. 

-----
## Installation
Install and activate the 'SELEC' environment in your desired directory with the following commands:

`git clone https://github.com/SZ-Jasper/SELEC.git`

`cd SELEC`

`conda env create -f environment.yml` 

`conda activate selec`

This enviroment contains the following packages: <br>
- jupyter
- pandas
- numpy
- scikit-learn
- pip
- pip:
  - plotly==5.6.0
  - streamlit

After acivating the SELEC enviroment, install packages with the following command:

`python setup.py install`

-----
## Organization
```
SELEC
-----
setup.py                  
environment.yml                          
examples/                 
|-batteryprepare.ipynb          
selec/
|-tests/
|-model/
| |-knn.py             

```


-----
## Battery Data
The dataset provided in SELEC is modified from cycling data provided by Sandia National Laboratory. <br>
The data was retrieved from the [Battery Archive](http://www.batteryarchive.org/) repo. <br>
To see how battery data was obtained, please see `data` directory. 

Citation for accompanying publication:

Yuliya Preger et al 2020 J. Electrochem. Soc. 167 120532

-----
## Graphical User Interface

<img src=https://github.com/SZ-Jasper/SELEC/blob/main/doc/Visual/selec_sidebar.JPG width=300 alt="selec gui sidebar with dropdown menus" p align="left">

The SELEC module uses Streamlit (https://streamlit.io/), an open-source Python-based GUI that runs on the user's browser.

To run the GUI from local files, enter the following command from a terminal: 

`cd selec`

`streamlit run selec.py`
 
This should bring up a series of URLS on the terminal, which can be copy and pasted into a web browser.

Upon entering the SELEC interface, there will be a side bar with a series of dropdown menus representing the various battery parameters the user can specify. Click on the desired parameters, and click the calculate button to start the predictive calculations. 


After finishing the calculations, the following plots will appear on the GUI, all with respect to cycle number:
* Charge capacity 
* Discharge capcity 
* Coulombic efficiency
* Energy efficiency
* Charge energy
* Discharge energy


The user can interact with these plots by:
* Zooming in/out of the image and panning 
* Hovering over a point for specific data
* Rotating the 3D plots
* Saving the figure as a png


-----

## Tests
Use the following command to run tests in `test.py`: 

`python -m unittest SELEC_test.py`

-----
