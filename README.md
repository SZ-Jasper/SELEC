# SELEC
*Description of SELEC program*

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
- Python packages listed in `environment.yml`

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