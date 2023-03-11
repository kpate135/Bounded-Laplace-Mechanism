import pandas as pd
# import csv
import numpy as np
from numpy.linalg import norm
import random
import math


# =====================================================================================================================================================
# This is a basic structure of the code that we are going to implement
# We would not use the built-in method in numpy to do Bounded Leplace Algorithm, instead we would implement it by using only math and random libraries
# This program expects a COVID dataset which you can find in our GitHub link. The dataset was originally found on Kaggle.
# =====================================================================================================================================================


# ============Data Handling=================
# Load dataset
df = pd.read_csv("Clinical-data.csv")
# Do Data Cleaning w/ Pandas library
df.dropna(inplace=True) # We could remove all missing/null values since we have a large dataset
# We can remove unnecessary columns as well, there are A LOT that we won't care about
# cols_to_keep = ['INSERT COLUMNS']
# df = df.drop(columns=df.columns.difference(cols_to_keep)) # This specifies columns we want to keep, and it drops the rest of the columns (Could be useful)
# =================END=======================


# =======Algorithem Implementation===========
# Define an Algorithm that calculates sensitivity
def calculate_sensitivity(target_query):
    return np.max(norm(target_query, axis=1))

# Define an Algorithm that finds Epsilon

def calculate_scale(sensitivity, privacy_level): #might need to pass in data set as well? 
    #We would need to calculate sensitivity base on a few method, we can find the max or min from the dataset and then do max - min.
    #sensitivity = ? #empty for now, waiting for data cleaning to be done
  
    return (sensitivity / privacy_level)

# Define Bounded Laplace Algorithm

def Bounded_Laplace_Algorithm(original_data, scale, bound): # FIX ME, What input does it need?
    #original_data represents the data prior to noise being introduced
    #the scale refers to the λ or exponential decay needed for the laplace mechanism
    #the bound indicates the appopriate output domain in which the noise can be introduced

    return 1    # FIX ME, obviously 

# ==============END=========================




# ======Apply Algorithm to Dataset========

# Extract the target row from the dataset

# Calculate the sensitivity of the query 
sensitivity = calculate_sensitivity() #Calculate and pass in the correct target row here

# Set the privacy level (epsilon) 
epsilon = 1.0

# Set the bound for the magnitude of the noise introduced
bound = 1.0

scale = calculate_scale(sensitivity, epsilon) # might need to update # FIX ME!

#Call Bounded_Laplace_Algorithm passing in original data, laplace scale, and bound
# =============END=========================
