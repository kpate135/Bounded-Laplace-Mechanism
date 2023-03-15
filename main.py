import random
import math
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.datasets import load_iris
from scipy.stats import laplace

# =====================================================================================================================================================
# This is a basic structure of the code that we are going to implement
# We would not use the built-in method in numpy to do Bounded Leplace Algorithm, instead we would implement it by using only math and random libraries
# =====================================================================================================================================================
print("\n")
print("\n")
print(" ====================================================1. Data Handling============================================\n")
# ====================================================1. Data Handling=================================================================================
# Load dataset
iris_data_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data") # alternative way to load dataset
df = pd.read_csv("iris.csv")
print ("df.head(): ")
print(df.head())
print("\n")
iris = load_iris()
original_data = iris.data
# ==================================The following are commented out because the dataset is very clean to begin with====================================
# Do Data Cleaning w/ Pandas library
# df.dropna(inplace=True) # We could remove all missing/null values since we have a large dataset
# cols_to_keep = ['INSERT COLUMNS']
# df = df.drop(columns=df.columns.difference(cols_to_keep)) # This specifies columns we want to keep, and it drops the rest of the columns (Could be useful)
# =================END of Data Handling================================================================================================================
print(" ========================================End of Data Handling======================================================\n")






# =======Algorithem Implementation===========


# Define an Algorithm that calculates sensitivity
def calculate_sensitivity(target_query):
    return np.max(norm(target_query, axis=1))

# Define an Algorithm that finds the scale for Laplace
def calculate_scale(sensitivity, privacy_level):
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
    
    #print("Mean :", np.mean(data[:, columnNum])) #debug msg
    return np.mean(data[:, columnNum])

# Define Bounded Laplace Algorithm
def Bounded_Laplace_Algorithm(original_data, loc, scale, lower, upper, flag): 
    #original_data represents the data prior to noise being introduced
    #the scale refers to the Î» or exponential decay needed for the laplace mechanism
    #the bound indicates the appropriate output domain in which the noise can be introduce
   
    if (flag==1): # We use the library, for comparison use
        noise = np.random.laplace(scale=scale) #TODO: replace the current method with a manual implementation of applying laplace
    else:
        mu = loc #assign mu to the location parameter

        b = scale #assign b to the scale parameter 

        # transform from [uniform distribution] into [Laplace distribution]
        uniform_transform = np.random.uniform(low=0.0, high=1.0, size=1) # Generate the random value uniformly distributed of size 1
        p = uniform_transform #assign p to to the uniform random value beteen 0 and 1 
        
        inverse_CDF_noise = mu - b * np.sign(p - 0.5) * np.log(1 - 2 * np.abs(p - 0.5))

        # this checks whether the bound parameter is a tuple or not. if tuple, it assigns bound to lower and upper value. if not tuple, it assigns it to both lower and upper
        if isinstance(bound, tuple):
             lower, upper = map(float, bound)
        else:
             lower = upper = float(bound)

        #apply the bounding restrictions to the new method
        bounded_noise = np.clip(inverse_CDF_noise, -lower, upper)

        noise = bounded_noise

    return original_data + noise

print("\n")
print("\n")
print(" ===========================2. BEGINNING OF Testing If Bounded_Laplace_Algorithm Works============================\n")
# ============================BEGINNING OF Testing If Bounded_Laplace_Algorithm Works========================================================
# The following method is to find the optimal parameters for the BLA.
testing_data = iris_data_df.values.tolist() # We want to perform testing on all rows
scale_testing_values = [0.1, 0.5, 1]
bounds_testing_values = [(0, 1), (0, 10), (-1, 1), (-0.5, 0.5)]

# Set None to begin our testing
best_so_far_result = None  
best_so_far_params = None

# Main loop for testing, this is a nested nested loop we would test all combination of testin value, epsilon, and bound to find a good combination.

test_data = calculate_mean(original_data, "sepal.length")

for scale in scale_testing_values:
    for bound in bounds_testing_values:
        lower, upper = bound #set lower and upper to be the current bound.
        result = Bounded_Laplace_Algorithm(test_data, loc=0, scale=scale, lower=lower, upper=upper, flag=2)  # apply function to the subset we just created
        print(f"Calculating: Scale: {scale}, Bound: {bound}, Result: {result}")  
            
        if best_so_far_result is None or np.min(result) < np.min(best_so_far_result):  # since we looking for the smallest value
            best_so_far_params = (scale, bound) # record down what loop id we in
            best_so_far_result = result   # update best so far
            print("New record! Best so far has been updated! ^")
                

# AT THE END # Print the best out of the best
print(f"The Best Parameters are: Scale: {best_so_far_params[0]}, Bound: {best_so_far_params[1]}, Result: {best_so_far_result}")
print("\n")
print(" ===============================END OF Testing If Bounded_Laplace_Algorithm Works==============================\n")
# ============================END OF Testing If Bounded_Laplace_Algorithm Works========================================================


print("\n")
print("\n")
print(" ================== 3. Determine Accuracy using train-test-split and Naive Bayes classification ==================\n")
# ========== Determine Accuracy using train-test-split and Naive Bayes classification ==========
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from scipy.stats import laplace
import matplotlib.pyplot as plt

# Load the iris dataset to X and y for training and testing split 
iris = load_iris()
X, y = iris.data, iris.target

# import train_test_split library from sklearn with a 80%/20% train/test-split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute the accuracy of the naive_bayes classifier on the original (non-perturbed) testing data
clf = GaussianNB()
clf.fit(X_train, y_train)
original_score = clf.score(X_test, y_test) #find the accuracy between the testing data and actual data

# Define the privacy budget (epsilon) values to test
epsilon_values = [8.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1, 0.01] #[0.1, 0.5, 1.0, 2.0, 5.0] #[1000, 100, 10, 1, 0.1, 0.01]

# Set test_sensitivity
test_sensitivity = 1.0

# Define the number of simulations (to calculate accuracy for)
num_simulations = 100

# Initialize a list to store the average accuracy for each epsilon value
mean_accuracy = []

# Train and test the naive Bayes classifier for each value of epsilon we want to test
for epsilon in epsilon_values:
    # Initialize a list to store the accuracy for each simulation
    accuracy_list = []
    
    for i in range(num_simulations):
        # Add Laplace noise to each feature's mean
        for j in range(X_train.shape[1]):
            X_train[:, j] += laplace.rvs(loc=0, scale=test_sensitivity/epsilon, size=X_train.shape[0]) #go through each column and take laplace for every row 
        
        # Fit the naive Bayes classifier on the perturbed training data
        clf = GaussianNB()
        clf.fit(X_train, y_train)
    
        # Evaluate the accuracy of the classifier on the testing data
        score = clf.score(X_test, y_test)
        accuracy_list.append(score)
    
    # Compute the mean accuracy over 100 simulations
    print(f"Epsilon: {epsilon:.1f}, Sensitivity: {test_sensitivity:.2f}, Accuracy: {np.mean(accuracy_list):.3f}")
    mean_accuracy.append(np.mean(accuracy_list))
    
# Plot the comparison accuracy vs epsilon for a differentially private naive Bayes classifer 
plt.plot(epsilon_values, mean_accuracy, '-.', label="Differentially Private")
plt.axhline(y=original_score, color='g', linestyle='--', label="Non-Private Baseline")
plt.legend(loc="upper left")
#plt.plot(epsilon_values, avg_accuracy, '-o')
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epsilon for Differentially Private Naive Bayes')
plt.show()
print(" ============================================END of Accuracy Test==================================================\n")


print("\n")
print(" ============================================4. Apply Algorithm to the (Whole) Dataset============================\n")
# ======Apply Algorithm to the (Whole) Dataset========

# Extract the target row from the dataset

# Calculate the sensitivity of the query 
sensitivity = calculate_sensitivity(original_data) #Calculate and pass in the correct target row here

# Set the location parameter 
loc = 0.0 

# Set the privacy level (epsilon) 
epsilon = best_so_far_params[0] # Use the best parameter that was computed earlier.

# Set the bound for the magnitude of the noise introduced
lower, upper = best_so_far_params[1] # Use the best parameter that was computed earlier.

 
# Set the scale parameter based on sensitivity and privacy level
scale = calculate_scale(sensitivity, epsilon)  

#Calculate the mean of the target column (the value we want to apply privacy to)
#define the columnName to calculate mean on
columnName = "sepal.length"
original_data = calculate_mean(original_data, columnName) #specify the column of the iris dataset

#Call Bounded_Laplace_Algorithm passing in original data, location parameter, laplace scale, bound, and flag (indicating which Laplace implementation to use)
noisy_data_np = Bounded_Laplace_Algorithm(original_data, loc, scale, lower, upper, 1)  # Using the numpy function.
noisy_data_CEK = Bounded_Laplace_Algorithm(original_data, loc, scale, lower, upper, 2) # Using the our BLA function.
#Compare original data and noisy data (after applying Bounded Laplace Mechanism)
print("Original Data: ", original_data)
print("Noisy Data with Numpy Implementation: ", noisy_data_np)
print("Noisy Data with CEK Implementation: ", noisy_data_CEK)
print(" ===================================================== END ======================================================\n")
# =============END OF MAIN=========================
