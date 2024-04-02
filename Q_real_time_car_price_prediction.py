import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("CSV/Volkswagen_Polo_gt_olx.csv")
# print(data)

df = pd.DataFrame(data)
# print(df.columns, df) 
##'Year', 'Price', 'Kilometers'
# print(df.isna().sum())

reg = LinearRegression()
reg.fit(df[["Year","Kilometers"]], df['Price'])


print("----- Only Volkswagen Polo -----")
print("Example:[Volkswagen Polo,2017,129000]")

year = int(input("Enter the year:- "))
KM = int(input("Enter the Kilometers:- "))

price_prediction  = reg.predict([[year,KM]])
predicted_price = math.floor(price_prediction)
print(predicted_price)

