
# coding: utf-8

# ###  Deploying Machine Learning Models

# In[66]:


#load  libraries

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import utils
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[67]:


#load iris datasets
iris_df = sns.load_dataset('iris')


# In[68]:


print(iris_df.shape)
print(iris_df.columns)


# In[69]:


iris_df.info()


# In[70]:


iris_df.sample(n=6)


# In[71]:


# descriptions
print(iris_df.describe())


# In[73]:


iris_df['species'] = iris_df['species'].map({'setosa': 0, 'virginica': 1,'versicolor': 2}).astype(int)


# In[74]:


iris_df.head()


# In[76]:


# histograms
iris_df.hist()
pyplot.show()


# In[77]:


# pairplot in order to see feature distribution among different samples

sns.pairplot(iris_df, kind="scatter", hue="species", markers=["o", "s", "D"], palette="Set2")
plt.show()


# In[78]:


# split data into training and testing data:


# In[79]:


y = iris_df["species"]
X = iris_df.drop("species",axis=1)


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_train = utils.to_categorical(y_train) 
y_test = utils.to_categorical(y_test)


# In[82]:


#create and compile the modelmodel = Sequential()
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[83]:


#Finally, we can train it:
model.fit(X_train, y_train, epochs=300, batch_size=10)


# In[84]:


# evaluate model:
scores = model.evaluate(X_test, y_test)
print("\nAccuracy: %.2f%%" % (scores[1]*100))


# In[86]:


#save  model in.into h5 for  web app  use    
model.save('model/model.h5')


# ### ResT  API 

# In[88]:


from flask import Flask, render_template, request, jsonify
import flask


# In[91]:


import tensorflow as tf
import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.models import load_model


app = Flask(__name__)

@app.before_first_request
def load_model_to_app():
    app.predictor = load_model('./static/model/model.h5')
    

@app.route("/")
def index():
    return render_template('index.html', pred = 0)

@app.route('/predict', methods=['POST'])
def predict():
    data = [request.form['spatial_length'],
            request.form['spatial_width'],
            request.form['petal_length'], 
            request.form['petal_width']]
    data = np.array([np.asarray(data, dtype=float)])

    predictions = app.predictor.predict(data)
    print('INFO Predictions: {}'.format(predictions))

    class_ = np.where(predictions == np.amax(predictions, axis=1))[1][0]

    return render_template('index.html', pred=class_)

def main():
    """Run the app."""
    app.run(host='0.0.0.0', port=8000, debug=False)  # nosec


if __name__ == '__main__':
    main()


# In[90]:




