# import csv
import random
import math
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.datasets import load_iris

# =====================================================================================================================================================
# This is a basic structure of the code that we are going to implement
# We would not use the built-in method in numpy to do Bounded Leplace Algorithm, instead we would implement it by using only math and random libraries
# This program expects a COVID dataset which you can find in our GitHub link. The dataset was originally found on Kaggle.
# =====================================================================================================================================================


# ============Data Handling=================
# Load dataset
df = pd.read_csv("iris.csv")
print(df.head)
iris = load_iris()
original_data = iris.data
# Do Data Cleaning w/ Pandas library
# df.dropna(inplace=True) # We could remove all missing/null values since we have a large dataset
# We can remove unnecessary columns as well, there are A LOT that we won't care about
# cols_to_keep = ['INSERT COLUMNS']
# df = df.drop(columns=df.columns.difference(cols_to_keep)) # This specifies columns we want to keep, and it drops the rest of the columns (Could be useful)
# =================END=======================







# =======Algorithem Implementation===========
# Define an Algorithm that calculates sensitivity
def calculate_sensitivity(target_query):
    return np.max(norm(target_query, axis=1))

# Define an Algorithm that finds the scale for Laplace

def calculate_scale(sensitivity, privacy_level): #might need to pass in data set as well? 
    #We would need to calculate sensitivity base on a few method, we can find the max or min from the dataset and then do max - min.
    #sensitivity = ? #empty for now, waiting for data cleaning to be done
  
    return (sensitivity / privacy_level)

# Define an Algorithm to calculate mean on a specified column

def calculate_mean(data, columnName):
    columnNum = 0
    if (columnName == "sepal.length"):
        columnNum = 0
    elif (columnName == "sepal.width"):
        columnNum = 1
    elif (columnName == "petal.length"):
        columnNum = 2
    elif (columnName == "petal.width"):
        columnNum = 3
    else:
        print("Invalid Column Name Entered")
        exit(0)
    
    #print("Mean :", np.mean(data[:, columnNum]))
    return np.mean(data[:, columnNum])

# Define Bounded Laplace Algorithm
# FIX ME
# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================
def Bounded_Laplace_Algorithm(original_data, loc, scale, bound, flag): # FIX ME, What input does it need?
    #original_data represents the data prior to noise being introduced
    #the scale refers to the Î» or exponential decay needed for the laplace mechanism
    #the bound indicates the appopriate output domain in which the noise can be introduced
    if (flag==1): # We use the library
        noise = np.random.laplace(scale=scale) #TODO: replace the current method with a manual implementation of applying laplace
    else:
        mu = loc #assign mu to the location parameter

        b = scale #assign b to the scale parameter 

        # transform from [uniform distribution] into [Laplace distribution]
        uniform_transform = np.random.uniform(low=0.0, high=1.0, size=1) # Generate the random value uniformly distributed of size 1
        p = uniform_transform #assign p to to the uniform random value beteen 0 and 1 
        
        inverse_CDF_noise = mu - b * np.sign(p - 0.5) * np.log(1 - 2 * np.abs(p - 0.5))

        #apply the bounding restrictions to the new method
        bounded_noise = np.clip(inverse_CDF_noise, -bound, bound)

        noise = bounded_noise

    return original_data + noise
# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================








# ======Apply Algorithm to Dataset========

# Extract the target row from the dataset

# Calculate the sensitivity of the query 
sensitivity = calculate_sensitivity(original_data) #Calculate and pass in the correct target row here

# Set the location parameter 
loc = 0.0

# Set the privacy level (epsilon) 
epsilon = 1.0

# Set the bound for the magnitude of the noise introduced
bound = 1.0

scale = calculate_scale(sensitivity, epsilon) # might need to update # FIX ME!

#Calculate the mean of the target column (the value we want to apply privacy to)
#define the columnName to calculate mean on
columnName = "sepal.length"
original_data = calculate_mean(original_data, columnName) #specify the column of the iris dataset

#Call Bounded_Laplace_Algorithm passing in original data, location parameter, laplace scale, bound, and flag (indicating which Laplace implementation to use)
noisy_data_np = Bounded_Laplace_Algorithm(original_data, loc, scale, bound, 1)
noisy_data_CEK = Bounded_Laplace_Algorithm(original_data, loc, scale, bound, 2)
#Compare original data and noisy data (after applying Bounded Laplace Mechanism)
print("Original Data: ", original_data)
print("Noisy Data with Numpy Implementation: ", noisy_data_np)
print("Noisy Data with CEK Implementation: ", noisy_data_CEK)
# =============END=========================
