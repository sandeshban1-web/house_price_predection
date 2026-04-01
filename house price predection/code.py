import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("house_data.csv")

df = pd.get_dummies(df, columns=['location'])
print("\nDataset:\n", df)

X = df.drop('price', axis=1)
y = df['price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("\nModel Accuracy (R2 Score):", r2_score(y_test,y_pred))

plt.scatter(df['area'], df['price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()

area = float(input("\nEnter area: "))
bedrooms = int(input("Enter bedrooms: "))
bathrooms = int(input("Enter bathrooms: "))
age = int(input("Enter age: "))
loc = input("Enter location (urban/suburban/rural): ").lower()

loc_urban = 1 if loc == "urban" else 0
loc_suburban = 1 if loc == "suburban" else 0
loc_rural = 1 if loc == "rural" else 0

input_data = [[area,bedrooms,bathrooms,age,loc_rural,loc_suburban,loc_urban]]

prediction = model.predict(input_data)

print("\nPredicted House Price:", prediction[0])
        
