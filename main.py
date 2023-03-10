import pandas as pd
# import csv
import random
import math

# =====================================================================================================================================================
# This is a basic structure of the code that we are going to implement
# We would not use the built-in method in numpy to do Bounded Leplace Algorithm, instead we would implement it by using only math and random libraries
# This program expects a COVID dataset which you can find in our GitHub link. The dataset was originally found on Kaggle.
# =====================================================================================================================================================


# ============Data Handling=================
# Load dataset
df = pd.read_csv('Clinical_data.csv')
# Do Data Cleaning
# CODE HERE
# =================END=======================


# =======Algorithem Implementation===========
# Define an Algorithm that finds Epsilon

def calculate_epsilon(privacy_level): #might need to pass in data set as well? 
    #We would need to calculate sensitivity base on a few method, we can find the max or min from the dataset and then do max - min.
    #sensitivity = ? #empty for now, waiting for data cleaning to be done
  
    return (sensitivity / privacy_level)

# Define Bounded Laplace Algorithm

def Bounded_Leplace_Algorithm(): # FIX ME, What input does it need?
    
    return 1    # FIX ME, obviously 

# ==============END=========================




# ======Apply Algorithm to Dataset========
privacy_level = 0.5 # User define value
epsilon = calculate_epsilon(privacy_level) # might need to update # FIX ME!
# =============END=========================
