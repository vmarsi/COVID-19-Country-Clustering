# Clustering European countries based on the associated contact patterns and socioeconomic indicators using epidemic 
# models

This repository was developed for thesis V. Marsi and Zs. Vizi "Clustering European countries based on the associated 
contact patterns and socioeconomic indicators using epidemic models", 2023.

## Install
This project is developed and tested with Python 3.10. To install project dependencies, execute the following steps:
1) Install *virtualenv*: `pip3 install virtualenv`
2) Create a virtual environment (named *venv*) in the project directory: `python3 -m venv venv`
3) Activate *venv*: `source venv/bin/activate`
4) Install dependencies from `requirements.txt`: `pip install -r requirements.txt`

## Data

- contact matrices for countries can be found in `./data/contact_x.xls` file, 
where x can be 'home', 'school', 'work', 'other'
- population vector for the European countries are located in `./data/age_data.xls`
- epidemic model parameters are listed in `./data/model_parameters.json`
- socioeconomic indicators are in `./data/indicators.xlsx`

## Framework
A summary about the steps of the procedure:
![alt text](https://drive.google.com/uc?export=download&id=1HFZl8GrZVtoYlXhA36HfYTHUfRj6BlmJ)

## Files
Code for running the framework is available in folder `./src`. 
Here you find
- the class for running the complete pipeline (`./src/main_run.py`)
- the data loading functionalities (`./src/dataloader.py`)
- the classes related to epidemic modelling (`./src/model.py`, `./src/r0.py`)
- the classes related to standardization (`./src/standardizer.py` and `./src/beta0.py`)
- the class for dimensionality reduction (`./src/dimension_reduction.py`)
- the class for executing clustering (`./src/clustering.py`)
- the utilities (`./src/utils.py`)

## Pipeline
Run `./src/main_run.py` as described below
- Concept of Standardizer is either "base_r0" or "final_death_rate". The initial values are base_r0=2.2 and 
final_death_rate=0.001. 
- The variable dim_red in DimRed can be "PCA" and "2D2PCA" depending on which technique we want to use.
- In Clustering we can change the name of the output files by changing img_prefix. We can cut the output dendrogram 
at a threshold using the variable threshold. The input variable dist is initially set to "Euclidean" but we can change 
it to "Manhattan" if we want to use Manhattan distance while clustering.

## Plots
If you run the code, the plots will be generated in the in-use created folder `./plots`.
