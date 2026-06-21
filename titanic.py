#Perform following tasks on the previously used Titanic Dataset -
#Make a new data frame out of the original dataset that consists of all numerical features only.
#Plot a boxplot for all the features of this new data frame
#Decide whether to normalize the data or to standardize
#Use Python Scikit-learn library to perform either Normalization (MinMaxScaler) or Standardization (StandardScaler)

import pandas as pd
import matplotlib.pyplot as plt

# Load the Titanic dataset
titanic_data = pd.read_csv('Titanic Dataset.csv')
# Create a new data frame with only numerical features
numerical_features = titanic_data.select_dtypes(include=['number'])
# Plot a boxplot for all the features of the new data frame
plt.figure(figsize=(12, 8))
numerical_features.boxplot()
plt.title('Boxplot of Numerical Features')
plt.xticks(rotation=45)
plt.show()
#decide whether to normalize or standardize the data
# In this case, we will standardize the data using StandardScaler
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
# Create an instance of StandardScaler
scaler = StandardScaler()
# Fit and transform the numerical features
scaled_features = scaler.fit_transform(numerical_features)
# Create a new data frame with the scaled features
scaled_numerical_features = pd.DataFrame(scaled_features, columns=numerical_features.columns)
# Display the first few rows of the scaled data frame
print(scaled_numerical_features.head())
