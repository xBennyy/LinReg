#import modules
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot 
import pickle 
from matplotlib import style

#import csv file
data = pd.read_csv("D:\Computer Science\programming\python\MLAI\student-mat.csv", sep=";")
#select data to use
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"
#define best variable
best = 0
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#allocating 10% of data to testing, and 90% to training to prevent getting reoccuring results
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
 
#training program for maximum accuracy
"""for _ in range (100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best :
        best=acc

        with open("studentmod.pickle", "wb") as f:
             pickle.dump(linear, f) """

#loading previously trained model, saved as studentmod.pickle
picklein = open("studentmod.pickle", "rb")
linear = pickle.load(picklein)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)
# print results
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

#plotting graph

p = "G1"

style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Predicted Grade")
pyplot.show()