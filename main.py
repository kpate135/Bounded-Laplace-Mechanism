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
# =====================================================================================================================================================


# ============Data Handling=================
# Load dataset
iris_data_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data") # alternative way to download the data. This is so everyone can run our code?
df = pd.read_csv("iris.csv")
print(df.head())
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
def Bounded_Laplace_Algorithm(original_data, loc, scale, bound, flag): 
    #original_data represents the data prior to noise being introduced
    #the scale refers to the Î» or exponential decay needed for the laplace mechanism
    #the bound indicates the appropriate output domain in which the noise can be introduced
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


# This is to test if the algorithem works as expected, with this dataset, that dataset, or... more dataset?
# ============================BEGINNING OF Testing If Bounded_Laplace_Algorithm Works========================================================
# The following method refers to the technique I learned in CS170 AI where I found the best so far by iterating through (lmao) all possible combinations!
testing_data = iris_data_df.values.tolist() # We want to perform testing on all rows
num_testing_values = [10, 50, 100, 500, 1000] # So our algorithem can be applied to different size of dataset (not only the iris dataset)
epsilon_testing_values = [0.1, 0.5, 1, 1.5, 2]
bounds_testing_values = [(0, 1), (0, 10), (0, 100), (-10, 10), (-100, 100)]

# Set None to begin our testing
best_so_far_result = None  
best_so_far_params = None

# Main loop for testing, this is a nested nested loop we would test all combination of testin value, epsilon, and bound to find a good combination.
for trial_number in num_testing_values:  # this loop can be removed if we are not going to apply this algorithem to another dataset, fun to keep for now, until it is taking too long to run
    for epsilon in epsilon_testing_values:
        for bound in bounds_testing_values:
            lower, upper = bound
            
            #sample subset data from orginal dataset using loop
            temp_data_subset = [] #empty out subset
            for row in testing_data:
                temp_row = [] #empty out temp_subset
                for x in range(len(row)):
                    temp_row.append(random.uniform(lower, upper))  # selecting random subset to ensure we are not overfitting 
                temp_data_subset.append(temp_row)
            #temp_data_subset now contains a subset of the original data with random values within the bounds of some features. 
            #btw, do we need this for this project scope?
            
            result = Bounded_Laplace_Algorithm(temp_data_subset, epsilon, lower, upper)  # apply function to the subset we just created
            # print(f"Num_testing_values: {trial_number}, Epsilon: {epsilon}, Bound: {bound}, Result: {result}")  #print out to see, too much message, this is comment out
            
            if best_so_far_result is None or result < best_so_far_result:  # since we looking for the smallest value
                best_so_far_params = (trial_number, epsilon, bound) # record down what loop id we in
                best_so_far_result = result   # update best so far
                print(f"Num_testing_values: {trial_number}, Epsilon: {epsilon}, Bound: {bound}, Result: {result}")  #print out to see ONLY when best so far updated
                

# AT THE END # Print the best out of the best
best_so_far_params.trial_number
print(f"Num_testing_values: {best_so_far_params.trial_number}, Epsilon: {best_so_far_params.epsilon}, Bound: {best_so_far_params.bound}, Result: {best_so_far_result}")  

# ============================END OF Testing If Bounded_Laplace_Algorithm Works========================================================




# ======Apply Algorithm to the (Whole) Dataset========

# Extract the target row from the dataset

# Calculate the sensitivity of the query 
sensitivity = calculate_sensitivity(original_data) #Calculate and pass in the correct target row here

# Set the location parameter 
loc = 0.0 

# Set the privacy level (epsilon) 
epsilon = 1.0 # The code above should find the optimal value for this already, double check if yes please fix me.

# Set the bound for the magnitude of the noise introduced
bound = 1.0 # The code above should find the optimal value for this already, double check if yes please fix me. FIX ME to a (x,y) value type as well
#Also need to fix the main def of the algorithem. Should be Ez.
 
# Set the scale parameter based on sensitivity and privacy level
scale = calculate_scale(sensitivity, epsilon) 

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
# =============END OF MAIN=========================



