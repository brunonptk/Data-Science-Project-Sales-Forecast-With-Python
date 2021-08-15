Our challenge is to be able to predict the sales that we will have in a given period based on spending on ads in the 3 major networks that a company invests: TV, Newspaper and Radio.
Database: https://drive.google.com/drive/folders/blablabla

Step by Step of a Data Science Project:
Step 1: Understanding the Challenge
Step 2: Understanding the Area/Company
Step 3: Data Extraction/Obtainment
Step 4: Data Adjustment (Treatment/Cleaning)
Step 5: Exploratory Analysis
Step 6: Modeling + Algorithms (This is where Artificial Intelligence comes in, if necessary)
Step 7: Interpretation of Results

Importing the Database:

import pandas as pd

chart = pd.read_csv("advertising.csv")
display(chart)

Exploratory Analysis:
Let's try to visualize how each item's information is distributed
Let's see the correlation between each of the items

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(chart.corr(), cmap="Wistia", annot=True)
plt.show()

sns.pairplot(chart)
plt.show()

With that, we can start preparing the data to train the Machine Learning Model. Separating into training data and test data.

from sklearn.model_selection import train_test_split 

# separete the information in X and Y

# y - it's what we want to find out
y = chart["Sales"]

# x - it's the rest of things
x = chart.drop("Sales", axis=1) # 0 to rows and 1 to columns

# apply the train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

We have a regression problem - Let's choose the models we're going to use:
Linear Regression
RandomForest (Decision Tree)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

model_linearregression = LinearRegression()
model_randomforest = RandomForestRegressor()

model_linearregression.fit(x_train, y_train)
model_randomforest.fit(x_train, y_train) 

AI Test and Best Model Evaluation
Let's use R² -> says the % that our model can explain what happens

forecast_linearregression = model_linearregression.predict(x_test)
forecast_randomforest = model_randomforest.predict(x_test)

from sklearn import metrics

# R2 -> from 0% to 100%, the higher, the better
print(metrics.r2_score(y_test, forecast_linearregression))
print(metrics.r2_score(y_test, forecast_randomforest))

# RandomForest is the best model
auxiliary_chart = pd.DataFrame()
auxiliary_chart["y_test"] = y_test
auxiliary_chart["linear regression"] = forecast_linearregression
auxiliary_chart["random forest"] = forecast_randomforest

plt.figure(figsize=(15, 5))
sns.lineplot(data=auxiliary_chart)
plt.show()

How important is each variable to sales?

sns.barplot(x=x_train.columns, y=model_randomforest.feature_importances_)
plt.show()

# import the new_chart with pandas (the new table must have TV, Radio and Newspaper data)
model_randomforest.predict(new_chart)
print(forecast)
