#dependencies import
import numpy as numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

#load data 
load_data = pd.read_csv ('X:\iris.csv')

#seperating the Class
x = load_data.values[:,0:3]
Y = load_data.values[:,4]

#Splitting the dataset into Test and Train
x_train, x_test,Y_train,Y_test = train_test_split(x,Y,test_size=0.2,random_state = 15)

#Function to perform training with Entropy
clf_entropy = DecisionTreeClassifier(criterion = 'entropy', random_state=15, max_depth = 7, min_samples_leaf = 3)

clf_entropy.fit(x_train,Y_train)

prediction = clf_entropy.predict(x_test)
print (("Prediction:", prediction))

print ("Accuracy : ", accuracy_score (Y_test,prediction))
