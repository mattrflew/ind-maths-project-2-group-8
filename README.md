# ind-maths-project-2-group-8

Group 8's repository for **Industrial Mathematics (MATH11231) - Project 2**

## Project 2 - Flocking Obstacles

See [Project Description](docs/8-Flocking-Obstacles-1.pdf)

## Explanation of repository

### run_tests Directory

This is where the test cases are run. There is a Jupyter Notebook to run each of the models. Notebooks read in the predefined test case configurations and execute the tests. Results are saved as .csv files to the _results_ directory.

### models Directory

This contains all of the files necessary to run each model. This includes:

#### Files

- **`params_default.py`**  
  Contains the default settings for simulations.

- **`initialise.py`**  
  Functions to initialise the placement of obstacles and flocks of birds in the simulations.

- **`functions.py`**  
  Common functions shared across all models.

#### Models

- **`model1.py`**  
  Extended Vicsek model.

- **`model2.py`**  
  Shill agent model. _(Note: This model may not run properly as effort was put on other models.)_

- **`model3.py`**  
  Steer to avoid model.

### test_cases Directory

Contains Jupyter Notebooks to generate combinations of parameters for defining various test cases. These parameter combinations are saved as .json files, which are later used to execute the corresponding tests.

### visual Directory

Contains Jupyter Notebooks which run the simulation and display an animation for each of the respective models.

### archive directory

Collection of Jupyter Notebooks to experiment with analysis methods and investigations for expanding the problem.

## Group 8

- Zhuohao Cai
- Matthew Flewelling
- Aleksandra Kalucka
- Juncheng Li
- Jessica Tunn
