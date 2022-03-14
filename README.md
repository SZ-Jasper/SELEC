# SELEC
*Description of SELEC program*

-----
## Graphical User Interface

The SELEC module uses streamlit (https://streamlit.io/), a Python-based GUI that runs on the user's browser. 

To use, the user must first install streamlit to their environment 
-->(will add directions to this later).

Then, run the following command on their terminal: streamlit run selec_gui.py
--> this will soon be changed to maybe the github link instead of a local py file
--> we can maybe even write a file that just does this for people? idk lol

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

This enviroment contrains the following packages:
-

-----
## Organization
*organization of git repository?*


-----
## Battery Data
*citation for battery data and link

-----
## Tests
Use the following command to run tests in `test.py`: 

`python -m unittest SELEC_test.py`

-----