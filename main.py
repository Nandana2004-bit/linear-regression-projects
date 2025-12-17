import pandas as pd
from sklearn .model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv(" battery_life.csv")
#print(df)

a = df.select_dtypes(include='number')
b = df.select_dtypes(include='object')
#print(a)
#print(b)
df.info()
df.describe()


#print(df.isnull())
df["network"]
c = {'WiFi': 0, 'MobileData' : 1}
df['encoded network'] = df['network'].map(c)
#print(df['network_mapped'])

X = df[["screen_time", "apps_used", "encoded network"]]
y = df[["battery_hours"]]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)