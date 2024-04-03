import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    'Area': [2600, 3000, 3200, 3600, 4000],
    'Price': [550000, 565000, 610000, 680000, 725000]
}

df = pd.DataFrame(data)
# print(df)
# print(df.columns)

#data plotting
plt.scatter(df["Area"], df["Price"], color='red', marker='+')
plt.xlabel("Area")
plt.ylabel("Price")
plt.savefig("LR-OP/LR(1_data_plotting).png")

#filtering linearregression model
reg = LinearRegression()
reg.fit(df[["Area"]],df["Price"]) #[[Independent Variable]],[Dependent Variable]  Independent Variable: Area    Dependent Variable: Price

# Plotting regression line
plt.scatter(df["Area"], df["Price"], color='red', marker='+')

plt.plot(df["Area"], reg.predict(df[["Area"]]), color = 'blue')

plt.xlabel("Area")
plt.ylabel("Price")
plt.savefig("LR-OP/LR(2_regression_line).png")

price = int(input("Enter the area to predict price: "))
val = reg.predict([[price]])

print("---------------------------------")
print("Slope (m):", reg.coef_)
print("---------------------------------")
print("Intercept (b):", reg.intercept_)
print("---------------------------------")
print("Predicted price:", val[0])

plt.plot([price], val, marker='o', markersize=8, color='green')  # Plotting the predicted price
plt.xlabel("Area")
plt.ylabel("Price")
plt.savefig("LR-OP/LR(3_predicted_price).png")
plt.show()
