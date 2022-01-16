# Conveyance  

**Problem Statement**  
When a delivery rider arrives at the customer-location that the customer entered for the order, the rider logs in our rider-app that they have arrived at the customer. However, because of any number of reasons (buildings blocking GPS signal, poor data entry by the customer, a rider’s newness, etc) the customer’s actual location might not match the customer’s entered location. As a result, the rider then starts the journey of finding their way from the customer’s entered location to the customer’s actual location. We track this traversal distance as dropoff_distance. In an ideal world, all dropoff_distance’s are zero.


**Folder Structure**  
data - all relevant and intermediate datasets inside  
model - all trained models  
src - python scripts for feature extraction and model training  
src/notebooks - notebooks containing analysis and training  
config.yaml - the goto file for all knobs of the project, all variables and parameters defined here  
controller.py - the entry point that processes and calls all other function  
requirements.txt - list of all packages needed to run the script  
Approach Brief pdf - This file contains all required details related to the assignment  

    

**Installation**  
The folder can be downloaded or cloned using git clone. 
Switch to virtual env and activate. 

source venv/bin/activate

requirements can be installed using:

pip3 install -r requirements.txt



**Execution**  
Paths and variables can be set inside config.yaml  
Steps to execute can be modified inside the config.yaml file  

Execute the code using
python3 controller.py  
