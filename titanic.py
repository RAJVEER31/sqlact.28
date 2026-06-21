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
#plot a  bar graph for age and survivors of the new data frame
age_survived = titanic_data.groupby('Age')['Survived'].sum()
plt.figure(figsize=(10, 6))
age_survived.plot(kind='bar')
plt.title('Bar Graph of Age and Survivors')
plt.xlabel('Age')
plt.ylabel('Number of Survivors')
plt.show()
#plot a pie chart for gender of the new data frame
gender_counts = titanic_data['Gender'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart of Gender')
plt.axis('equal')
plt.show()
#plot a fair plot for age of the new data frame
plt.figure(figsize=(10, 6))
plt.hist(titanic_data['Age'], bins=20, edgecolor='black')
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
#plot a categorical plot for survived of the new data frame
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=titanic_data)
plt.title('Count Plot of Survived')
plt.xlabel('Survived')
plt.ylabel('Count')
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
