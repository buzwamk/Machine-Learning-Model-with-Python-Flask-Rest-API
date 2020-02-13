#!/usr/bin/env python
# coding: utf-8

# ### Boston House Prices dataset
# 

# This dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass
# ######  There are 14 attributes in each case of the dataset. They are:
# - CRIM - per capita crime rate by town
# - ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# - INDUS - proportion of non-retail business acres per town.
# - CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# - NOX - nitric oxides concentration (parts per 10 million)
# - RM - average number of rooms per dwelling
# - AGE - proportion of owner-occupied units built prior to 1940
# - DIS - weighted distances to five Boston employment centres
# - RAD - index of accessibility to radial highways
# - TAX - full-value property-tax rate per
# - PTRATIO - pupil-teacher ratio by town
# - B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# - LSTAT - % lower status of the population
# - MEDV - Median value of owner-occupied homes in $1000's


# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split

# importing dataset from sklearn
from sklearn.datasets import load_boston
boston_data = load_boston()



# initializing dataset
data_df = pd.DataFrame(boston_data.data)

data_df.head()



# Adding features names to the dataframe
data_df.columns = boston_data.feature_names
data_df.head(10)


data_df.shape


# checking null values
data_df.isnull().sum()



# checking if values are categorical or not
data_df.info()


# lets  define y as the labels of the dataset. The labels in this case are the prices


# Target feature of Boston Housing data
#PRICE: the predicted price of a house based on its features.
data_df['PRICE'] = boston_data.target


data_df.head()

# visualize the relationship between the features and the response using scatterplots# visua 
fig, axs  =  plt.subplots(1,3,sharey=True)
data_df.plot(kind='scatter',x='CRIM',y='PRICE',ax=axs[0],figsize=(16,8))
data_df.plot(kind='scatter',x='LSTAT',y='PRICE',ax=axs[1])
data_df.plot(kind='scatter',x='AGE',y='PRICE',ax=axs[2])


# Correlation measures the strength of the linear relationship between two independent variables

sns.set(style="white")

# Compute the correlation matrix
corr = data_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool) # Return an array of zeros with the same shape and type as a given array
mask[np.triu_indices_from(mask)] = True # Return the indices for the upper-triangle of arr

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# cell above shows  correlation  between  features  

# ### Creating model
# At first we will need to separate feature and target variable. Then split data set into training and testing set. And finally create a model.


# creating feature and target variable 
X = data_df.drop(['PRICE'], axis=1)
y = data_df['PRICE'] 

# splitting into training and testing set

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
print("X training shape : ", X_train.shape )
print("X test shape : ", X_test.shape )
print("y training shape :" , y_train.shape )
print("y test shape :", y_test.shape )

# creating model
from sklearn.ensemble import RandomForestRegressor
classifier = RandomForestRegressor()
classifier.fit(X_train, y_train)


# evaluate the performance of model for training and testing set.

# Model evaluation for training data
prediction = classifier.predict(X_train)
print("r^2 : ", metrics.r2_score(y_train, prediction))
print("Mean Absolute Error: ", metrics.mean_absolute_error(y_train, prediction))
print("Mean Squared Error: ", metrics.mean_squared_error(y_train, prediction))
print("Root Mean Squared Error : ", np.sqrt(metrics.mean_squared_error(y_train, prediction)))

# Model evaluation for testing data
prediction_test = classifier.predict(X_test)
print("r^2 : ", metrics.r2_score(y_test, prediction_test))
print("Mean Absolute Error : ", metrics.mean_absolute_error(y_test, prediction_test))
print("Mean Squared Error : ", metrics.mean_squared_error(y_test, prediction_test))
print("Root Mean Absolute Error : ", np.sqrt(metrics.mean_squared_error(y_test, prediction_test)))

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(y_test, prediction, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], 
        [y_test.min(), y_test.max()], 
        'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

mse = mean_squared_error(y_test, prediction)
mse

classifier.score(X_test, y_test)

# ### Saving and using machine learning model

# saving the model
import pickle
with open('model/model.pkl','wb') as file:
    pickle.dump(classifier, file)

# saving the columns
model_columns = list(X.columns)
with open('model/model_columns.pkl','wb') as file:
    pickle.dump(model_columns, file)


# ### Creating API for machine learning using Flask

from flask import render_template, request, jsonify
import flask
import numpy as np
import traceback
import pickle
import pandas as pd
 
 
# App definition
app = Flask(__name__,template_folder='templates')
 
# importing models
with open('model/model.pkl', 'rb') as f:
   classifier = pickle.load (f)
 
with open('model/model_columns.pkl', 'rb') as f:
   model_columns = pickle.load (f)
 
 
@app.route('/')
def welcome():
   return "Boston Housing Price Prediction"
 
@app.route('/predict', methods=['POST','GET'])
def predict():
  
   if flask.request.method == 'GET':
       return "Prediction page"
 
   if flask.request.method == 'POST':
       try:
           json_ = request.json
           print(json_)
           query_ = pd.get_dummies(pd.DataFrame(json_))
           query = query_.reindex(columns = model_columns, fill_value= 0)
           prediction = list(classifier.predict(query))
 
           return jsonify({
               "prediction":str(prediction)
           })
 
       except:
           return jsonify({
               "trace": traceback.format_exc()
               })

if __name__ == "__main__":
   app.run()







