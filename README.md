# SELEC
The SELEC project (Maching Learning for Selecting Electrolytes) is aiming to produce a ML based program that can return the optimized electrolyte for a given electrode system. Considering the continuously growring research interest in battery industry, tons of battery tests are being generated and thus numerous battery datasets are available for machine learning study. In this project, the program is built up with a regression model (categorize & predict data) and a graphical user interface (for electrode determination & output display). The model will be trained with a given battery dataset that involves information of electrodes, electrolytes, battery performances in different aspects. The trained model will then be validated and tested with spliited dataset and the final prediction will be stored in a local file. The users are thus able to access the GUI by specifying battery parameters of interest and the program will returen the best combination based on the prediction result from the previously trained dataset. 

-----
## Graphical User Interface

The SELEC module uses streamlit (https://streamlit.io/), a Python-based GUI that runs on the user's browser. 

*Link to GUI*

Then, run the following command on their terminal: 

`streamlit run selec_gui.py`

--> this will soon be changed to maybe the github link instead of a local py file

*Tutorial -> image of how to input -> image of resulting plots

-----
## Software Dependencies
The SELEC predictor is available on the web and python is not required. Please see *Interface Instructions* for more information. 
For those who would like to run the jupyter and python files, please ensure you have the following:
- Python 3.7
- Python packages listed in `environment.yml`and Installation section. 

-----
## Installation
Install and activate the environment with `environment.yml` with the following commands:

`conda env create -f environment.yml` 

`conda activate SELEC_env` 

This enviroment contains the following packages: <br>
-numpy

-----
## Organization
*organization of git repository?*


-----
## Battery Data
The dataset provided in SELEC is modified from cycling data provided by Sandia National Laboratory. 

The data was retrieved from the [Battery Archive](http://www.batteryarchive.org/). 

Citation for accompanying publication:

Yuliya Preger et al 2020 J. Electrochem. Soc. 167 120532

-----
## Tests
Use the following command to run tests in `test.py`: 

`python -m unittest SELEC_test.py`

-----
