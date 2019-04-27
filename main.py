import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('USA_Housing.csv')

print(sns.pairplot((df)))
print(sns.distplot(df['Price']))

print(df.columns)

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms', 'Area Population']]

y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# create an instance of the linear regression model
lm = LinearRegression()

lm.fit(X_train, y_train)

print(lm.intercept_)

plt.show()
