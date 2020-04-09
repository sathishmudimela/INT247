import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("petrol_consumption.csv")
dataset.head()
dataset.describe()

x = dataset[['Petrol_tax','Average_income','Paved_Highways','Population_Driver_licence(%)']]
y = dataset['Petrol_Consumption']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
print("\nIntercept : ",reg.intercept_)
print("\nCoefficient : ",reg.coef_)

coeff_df = pd.DataFrame(reg.coef_,x.columns,columns=['Coefficient'])
print("\nCoefficient Difference : \n",coeff_df)
y_pred = reg.predict(x_test)
df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(df)

from sklearn import metrics
print("\n MeanAbsoluteError : ",metrics.mean_absolute_error(y_test,y_pred))
print("\n MeanSquaredError : ",metrics.mean_squared_error(y_test,y_pred))
print("\n RootMeanSquaredError : ",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
