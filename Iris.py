import pandas as pd
from sklearn.model_selection import train_test_split

# Create a list of column names
columnNames = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

# Create a DataFrame
dataset = pd.DataFrame(columns=columnNames)

# Add rows to the DataFrame
dataset.loc[0] = [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']
dataset.loc[1] = [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']
dataset.loc[2] = [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']
dataset.loc[3] = [4.6, 3.1, 1.5, 0.2, 'Iris-setosa']
dataset.loc[4] = [5.0, 3.6, 1.4, 0.2, 'Iris-setosa']

# Select the feature columns
feature_columns = ['SepalLength', 'SepalWidth', 'PetalLength','PetalWidth']

# Extract the feature values
X = dataset[feature_columns].values

# Select the target column
target_column = 'Species'

# Extract the target values
y = dataset[target_column].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print(X_train)
print(X_test)
print(y_train)
print(y_test)
