import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("house_rent.csv")
#print(df.head())

#02
df.shape
numeric_cols_df = df.select_dtypes(include=np.number)
#print(numeric_cols_df.columns)
df.describe()

#03
#print(df.isnull())
df["age"].fillna(df["age"].mean(), inplace=True)
#print(df.isnull())
#print(df.duplicated())

#04
X = df[["size_sqft", "bedrooms", "age"]]
Y = df[["rent"]]

#05
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#06
model = LinearRegression()
model.fit(X_train,Y_train)

#07
y_pred = model.predict(X_test)
print(y_pred)


