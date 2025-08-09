import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('anemia.csv')
df.head()
df.info()
df.shape
df.isnull().sum()
results = df['Result'].value_counts()
results.plot(kind = 'bar',color=['blue','green'])
plt.xlabel('Result')
plt.ylabel('Frequency')
plt.title('Count of Result')
plt.show()
# Import resample for undersampling
from sklearn.utils import resample


# Separate majority and minority classes
majorclass = df[df['Result'] == 0]
minorclass = df[df['Result'] == 1]

# Downsample majority class
major_downsample = resample(
    majorclass,
    replace=False,                 # sample without replacement
    n_samples=len(minorclass),    # match minority class count
    random_state=42               # reproducibility
)

# Combine minority class with downsampled majority class
df = pd.concat([major_downsample, minorclass])

# Check the class distribution
print(df['Result'].value_counts())

import matplotlib.pyplot as plt

result_balanced = df['Result'].value_counts()
result_balanced.plot(kind='bar', color=['blue', 'green'])
plt.xlabel('Result')
plt.ylabel('Frequency')
plt.title('Count of Result (Balanced)')
plt.show()

df.describe()

output = df['Gender'].value_counts()
output.plot(kind = 'bar', color =['orange','green'])
plt.xlabel('gender')
plt.ylabel('Frequency')
plt.title(' Gender count')
plt.show()

sns.displot(df['Hemoglobin'], kde=True)


plt.figure(figsize=(6, 6))

# Create the bar plot
ax = sns.barplot(
    x='Gender',
    y='Hemoglobin',
    hue='Result',
    data=df,
    ci=None
)

# Set the x-axis labels manually (optional)
ax.set_xticklabels(['Male', 'Female'])

# Add value labels on top of each bar
for container in ax.containers:
    ax.bar_label(container)

# Set plot title
plt.title("Mean Hemoglobin by Gender and Result")

# Show the plot
plt.show()

sns.pairplot(df)


# Compute the correlation matrix
correlation_matrix = df.corr()

# Set the figure size
fig = plt.gcf()
fig.set_size_inches(10, 8)

# Plot the heatmap
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='RdYlGn',
    linewidths=0.2
)

# Display the plot
plt.show()

x = df.drop('Result', axis = 1)
y = df['Result']

from sklearn.model_selection import train_test_split

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y,                # Features and labels
    test_size=0.2,       # 20% for testing
    random_state=20      # For reproducibility
)

# Print the shapes of each split
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create the model
logistic_regression = LogisticRegression()

# Step 2: Train the model on training data
logistic_regression.fit(x_train, y_train)

# Step 3: Predict on test data
y_pred = logistic_regression.predict(x_test)

# Step 4: Evaluate the model
acc_lr = accuracy_score(y_test, y_pred)
c_lr = classification_report(y_test, y_pred)

# Step 5: Print results
print('Accuracy Score:', acc_lr)
print(c_lr)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create the model
random_forest = RandomForestClassifier()

# Step 2: Train the model
random_forest.fit(x_train, y_train)

# Step 3: Predict on test data
y_pred = random_forest.predict(x_test)

# Step 4: Evaluate the model
acc_rf = accuracy_score(y_test, y_pred)
c_rf = classification_report(y_test, y_pred)

# Step 5: Print results
print('Accuracy Score:', acc_rf)
print(c_rf)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Initialize the model
decision_tree_model = DecisionTreeClassifier()

# Step 2: Train the model
decision_tree_model.fit(x_train, y_train)

# Step 3: Make predictions
y_pred = decision_tree_model.predict(x_test)

# Step 4: Evaluate the model
acc_dt = accuracy_score(y_test, y_pred)
c_dt = classification_report(y_test, y_pred)

# Step 5: Output the results
print('Accuracy Score:', acc_dt)
print(c_dt)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Initialize the model
NB = GaussianNB()

# Step 2: Train the model
NB.fit(x_train, y_train)

# Step 3: Make predictions
y_pred = NB.predict(x_test)

# Step 4: Evaluate the model
acc_nb = accuracy_score(y_test, y_pred)
c_nb = classification_report(y_test, y_pred)

# Step 5: Output the results
print('Accuracy Score:', acc_nb)
print(c_nb)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Initialize the model
support_vector = SVC()

# Step 2: Train the model
support_vector.fit(x_train, y_train)

# Step 3: Predict
y_pred = support_vector.predict(x_test)

# Step 4: Evaluate
acc_svc = accuracy_score(y_test, y_pred)
c_svc = classification_report(y_test, y_pred)

# Step 5: Display results
print('Accuracy Score:', acc_svc)
print(c_svc)

# Import necessary libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load sample dataset (Iris)
data = load_iris()
X = data.data
y = data.target

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Classifier
GBC = GradientBoostingClassifier()
GBC.fit(x_train, y_train)

# Predict on the test data
y_pred = GBC.predict(x_test)

# Evaluate the model
acc_gbc = accuracy_score(y_test, y_pred)
c_gbc = classification_report(y_test, y_pred)

# Print results
print('Accuracy Score:', acc_gbc)
print('Classification Report:\n', c_gbc)

# Sample new patient data (features)
new_data = [[0, 11.6, 22.3, 30.9, 74.5]]  # Replace these values with actual feature inputs

# Predict using the trained model
prediction = GBC.predict(new_data)

# Interpret and print the result
if prediction[0] == 0:
    print("You don't have any Anemic Disease")
elif prediction[0] == 1:
    print("You have anemic disease")

model = pd.DataFrame({
    'Model': [
        'Linear Regression', 'Decision Tree Classifier', 'RandomForest Classifier',
        'Gaussian Naive Bayes', 'Support Vector Classifier', 'Gradient Boost Classifier'
    ],
    'Score': [acc_lr, acc_dt, acc_rf, acc_nb, acc_svc, acc_gbc]
})

import pickle
import warnings

pickle.dump(GBC, open("model.pkl", "wb"))
