# SELEC

<img src=https://github.com/SZ-Jasper/SELEC/blob/main/doc/SELEC%20Logo.png width=300 p align="right">

The SELEC project (Maching Learning for Selecting Electrolytes) aims to produce a machine learning program that can return the optimized electrolyte for a given battery system. Considering the continuously growing research interest in the battery industry, tons of battery experiments are being conducted, and thus numerous battery datasets are available for machine learning study. In this project, the program is built up with a regression model (categorize & predict data) and a graphical user interface (for electrode determination & output display). The model will be trained with a given battery dataset that involves information of electrodes, electrolytes, battery performances in different aspects. The trained model was validated and tested with spliited dataset and the final prediction will be stored in a local file. The users are thus able to access the GUI by specifying battery parameters of interest and the program will returen the best combination based on the prediction result from the previously trained dataset. 


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

-----
## Organization
```
SELEC
-----              
environment.yml
.gitignore 
LICENSE
README.md
examples/                 
|-batteryprepare.ipynb
|-dataprep_example.ipynb
|-gui_example.ipynb
|-knn_example.ipynb
selec/
|-tests/
  |-__init__.py
  |-test_selec.py
|-model/
  |-knn.py  
  |-__init__.py
|-dataprepare/
  |-dataprep.py
  |-__init__.py
|-predictor/
  |-batterypredict.py
  |-__init__.py
|-__init__.py
|-ohe.obj
|-selec.py
supplementary/
|-Artificial_Neural_Network.ipynb
|-Decision_Trees_Regression.ipynb
|-Gaussian_Process_Regression.ipynb 
|-Gradient_Boosting_Machine.ipynb
|-KNN_Regression.ipynb
|-Neural Network Example.ipynb
|-Random_Forest_Regression.ipynb 
|-Support_Vector_Machine_Regression.ipynb
doc/
|-Visual/
  |-3D graph.ipynb
  |-selec_sidebar.JPG
  |-SELEC 2DPlot
  |-SELEC 3DPlot
|-SELEC Logo.png
|-SELEC_Poster.pptx
|-Technology Review Machine Learning.pptx
|-Technology Review User Interface.pptx
|-Use Cases and Component Specification.docx
data/
|-Battery_Dataset.csv
|-Battery_Dataset.ipynb
|-Cycle Data.zip
|-snl_metadata_cycle_500.csv
.github/workflows/
|-python-package-conda.yml
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

<img src=https://github.com/SZ-Jasper/SELEC/blob/main/doc/Visual/selec_sidebar.JPG width=320 alt="selec gui sidebar with dropdown menus" p align="left">

The SELEC module uses [Streamlit](https://streamlit.io/), an open-source Python-based GUI that runs on the user's browser.

To run the GUI from local files, follow the instructions in the "Installation" section, then enter the following command from a terminal: 

`cd selec`

`streamlit run selec.py`
 
This will return series of URLS on the terminal, which can be copy and pasted into a web browser.

Upon entering the SELEC interface, there will be a side bar with a series of dropdown menus representing the various battery parameters the user can specify. Click on the desired parameters, and click the calculate button to start the predictive calculations. 


After finishing calculations, the following plots will appear on the GUI, all with respect to cycle number:
* Charge capacity 
* Discharge capcity 
* Charge energy
* Discharge energy
* Coulombic efficiency
* Energy efficiency


The user can interact with these plots by:
* Zooming in/out of the image and panning 
* Hovering over a point for specific data
* Rotating the 3D plots
* Saving the figure as a png

<img src=https://github.com/SZ-Jasper/SELEC/blob/main/doc/Visual/SELEC%203DPlot.png width=800 p align = "center">

-----

## Tests
`SELEC.py` contains all functions from modules in `selec` directory <br>
Change into tests directory and Use the following command to run tests in `SELEC_test.py`:

`python -m unittest SELEC_test.py`

-----

## Authors
Karen Li, Chemical Engineering <br>
Rose Lee, Chemical Engineering <br>
Bella Wu, Material Science and Engineering <br>
Shuyan Zhao, Material Science and Engineering <br>
