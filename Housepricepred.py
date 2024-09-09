import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df=pd.read_csv("C:/Users/BLAUPLUG/Documents/Python_programs/House Price Prediction/kc_house_data.csv")
#print(df)

df=df.drop(['id','date'],axis=1)
df=pd.get_dummies(df)

df=df.fillna(df.mean())
#print(df)

X=df.drop(['price'],axis=1)
#print(X)

Y=df['price']
#print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)

MAE=metrics.mean_absolute_error(Y_test,y_pred)  #Avg abs difference between the predicted value and the actual value
print(MAE)


new_home=pd.DataFrame({'bedrooms':[3],'bathrooms':[1],'sqft_living':[1180],"sqft_lot":[5650],"floors":[1],'waterfront':[0],'view':[0],'condition':[3],'grade':[7],'sqft_above':[1180],'sqft_basement':[0],'yr_built':[1955],'yr_renovated':[0],"zipcode":[98178],"lat":[47.5712],"long":[-122.257],"sqft_living15":[1340],"sqft_lot15":[5650]})
new_home_pred=model.predict(new_home)

print("Prediction house price: ${:.2f}".format(new_home_pred[0]))
