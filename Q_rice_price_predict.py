import math
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression


data = pd.read_csv("CSV/vegetable_rice_data.csv")
# print(data)

df = pd.DataFrame(data)
# print(df.columns,df)
#['Petrol Charge ($)', 'Road Tax ($)', 'Kilometers Traveled','Price ($)']

# print(df.isna().sum()) #to find the total NaN values
# Petrol Charge ($)      0
# Road Tax ($)           1
# Kilometers Traveled    0
# Price ($)              1

#rename the column
df.rename(columns={'Petrol Charge ($)': 'Petrol_Charge'}, inplace=True)
df.rename(columns={'Road Tax ($)': 'Road_Tax'}, inplace=True)
df.rename(columns={'Kilometers Traveled': 'Kilometers_Traveled'}, inplace=True)
df.rename(columns={'Price ($)': 'Price'}, inplace=True)
# print(df.columns)
# ['Petrol_Charge', 'Road_Tax', 'Kilometers_Traveled', 'Price']

median_road_tax = math.floor(df.Road_Tax.median())
# print(median_road_tax) #137
median_price = math.floor(df.Price.median())
# print(median_price) #627
df.Road_Tax = df.Road_Tax.fillna(median_road_tax)
df.Price = df.Price.fillna(median_price)
# print(df)

reg = LinearRegression()
reg.fit(df[["Petrol_Charge","Road_Tax","Kilometers_Traveled"]], df['Price'])

print("______PRICE PREDICTION OF RICE______")
PC = float(input("Enter the Petrol price:- "))
RT = float(input("Enter the Road tax:- "))
KMT = float(input("Enter the Kilometers travelled:- "))

price_prediction = reg.predict([[PC,RT,KMT]])
predicted_price = math.floor(price_prediction)
print(f"The approximate price is {predicted_price}")