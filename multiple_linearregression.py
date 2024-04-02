import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
  "Area": [2600, 3000, 3200, 3600, 4000],
  "Bedrooms": [3, 4, None, 3, 5],
  "Age": [20, 15, 18, 30, 8],
  "Price": [550000, 565000, 610000, 595000, 760000]
}

df = pd.DataFrame(data)
# print(df)
# print(df.columns)

#data plotting
plt.scatter(df['Area'],df["Price"], color='red', marker='+')
plt.xlabel("Area")
plt.ylabel("Price")
plt.savefig("MLR-OP/1_MLR(data_plotting).png")

#finding the NAN value using median
median_bedroom = df.Bedrooms.median()
#round of the value using matgh.floor
median_bedroom = math.floor(df.Bedrooms.median())
# print(median_bedroom)

#filling the missing value (NAN) with median value
df.Bedrooms = df.Bedrooms.fillna(median_bedroom)
# print(df)

#Linearregression
reg = LinearRegression()
reg.fit(df[["Area","Bedrooms","Age"]],df["Price"]) #[[dependent variable]],[independent variable]

#plotting regression line
plt.scatter(df["Area"],df["Price"], color='red', marker='+')

plt.plot(df["Area"], reg.predict(df[["Area","Bedrooms","Age"]]), color='blue')

plt.xlabel("Area")
plt.ylabel("Price")
plt.savefig("MLR-OP/2_MLR(regression_line).png")


print("PRICE PREDICTION OF A HOUSE")
print("----------------------------------------------------------------")
AREA = int(input("Enter the Area:- "))
BEDROOMS = int(input("Enter the number of bedrooms:- "))
AGE = int(input("Enter the age:- "))

#prediction
price_prediction = reg.predict([[AREA,BEDROOMS,AGE]])
print(f"The approximate price will be:- {price_prediction}")
print("----------------------------------------------------------------")
print(reg.coef_)
print("----------------------------------------------------------------")
print(reg.intercept_)

#plotting predicted line
plt.plot([AREA], price_prediction, marker='o', markersize=8, color='green')
plt.xlabel("Area")
plt.ylabel("Price")
plt.savefig("MLR-OP/3_MLR(predicted_price).png")
plt.show()

# # Plotting predicted point
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df["Area"], df["Bedrooms"], df["Age"], c='red', marker='+')
# ax.scatter(AREA, BEDROOMS, AGE, c='green', marker='o', s=200)  # Plotting the predicted point
# ax.set_xlabel('Area')
# ax.set_ylabel('Bedrooms')
# ax.set_zlabel('Age')
# ax.set_title('Price Prediction')
# plt.savefig("MLR-OP/3_MLR(predicted_price).png")
# plt.show()