# SOIL_QUALITY_PREDICTION
  import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv("crop.csv")
df.head()

df.info()

df.describe()

df.isnull().sum()

sns.histplot(df["Fertilizer_Usage"])
plt.plot()

df.drop(columns=["Link"], inplace=True)
df.head()

# perform label encoding:
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()
for i in ["District_Name", "Soil_color", "Crop", "Fertilizer"]:
 df[i]=le.fit_transform(df[i])
df.head()

# split features and target variable
x=df.drop(columns=["Fertilizer_Usage"])
y=df["Fertilizer_Usage"]

# split the data into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# build the Linear Regression model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

# predict the xtest and store it into ypred
ypred=lr.predict(x_test)

# check the mae,mse and r2_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
mae = mean_absolute_error(y_test, ypred)
mse = mean_squared_error(y_test, ypred)
r2 = r2_score(y_test, ypred)


# Print results
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

