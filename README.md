# prediction_using_supervised-learning_ML-1

DATA SCIENCE & BUSINESS ANALYTICS
DATASET Link : http://bit.ly/w-data



topic:- Predict the percentage of an student based on the no of study hours.

Import the DataSet
Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#load DataSet from the Source
DataSetUrl = "http://bit.ly/w-data"
#reading the dataset
data = pd.read_csv(DataSetUrl)
data.head()
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25 entries, 0 to 24
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Hours   25 non-null     float64
 1   Scores  25 non-null     int64  
dtypes: float64(1), int64(1)
memory usage: 528.0 bytes
%matplotlib inline
data.plot(x="Hours",y="Scores",style='o')
plt.title("Hours vs Scores")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

Preaparing data for training and testing
x=data.iloc[:,:-1].values
y=data.iloc[:,1:].values
print(x)
[[2.5]
 [5.1]
 [3.2]
 [8.5]
 [3.5]
 [1.5]
 [9.2]
 [5.5]
 [8.3]
 [2.7]
 [7.7]
 [5.9]
 [4.5]
 [3.3]
 [1.1]
 [8.9]
 [2.5]
 [1.9]
 [6.1]
 [7.4]
 [2.7]
 [4.8]
 [3.8]
 [6.9]
 [7.8]]
print(y)
[[21]
 [47]
 [27]
 [75]
 [30]
 [20]
 [88]
 [60]
 [81]
 [25]
 [85]
 [62]
 [41]
 [42]
 [17]
 [95]
 [30]
 [24]
 [67]
 [69]
 [30]
 [54]
 [35]
 [76]
 [86]]
#now divided the data for training and testing the model
#import the train_test_split
from sklearn.model_selection import train_test_split

#splitting the data into x_train ,x_test, y_train, y_test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
Training the Model
#import the LinearRegression
from sklearn.linear_model import LinearRegression

#creating an object for LinearRegression
regression=LinearRegression()

#fitting the model
regression.fit(x_train,y_train)
LinearRegression()
#plotting the regression line
line = regression.coef_*x+regression.intercept_

#plotting for test data
plt.scatter(x,y)
plt.plot(x,line)
plt.show()

Checking Prediction
y_pred=regression.predict(x_test)
df=pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
df
Actual	Predicted
0	85	77.687022
1	47	52.551773
2	27	34.183707
3	24	21.616083
4	81	83.487464
Evaluting the Model
from sklearn.metrics import mean_absolute_error
rmse=mean_absolute_error(y_test,y_pred)
print("The Mean Absolute Error:",rmse)
The Mean Absolute Error: 4.983967953948152
Testing for the value
hours = np.array([9.25])
hours = hours.reshape(-1,1)
own_pred = regression.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
No of Hours = [[9.25]]
Predicted Score = [92.67149722]
