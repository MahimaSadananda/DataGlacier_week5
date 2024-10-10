# Importing the Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Reading the data
titanic = pd.read_csv('/content/titanic.csv')

# handling missing values
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)

# Splitting X and y
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

# logistic regressor
logistic_regressor = LogisticRegression()

# Fitting the model
logistic_regressor.fit(X, y)

# Saving model to disk
pickle.dump(logistic_regressor, open('titanic_model.pkl', 'wb'))

# Loading the model back to compare results
model = pickle.load(open('titanic_model.pkl', 'rb'))
print(model.predict(X[:5]))