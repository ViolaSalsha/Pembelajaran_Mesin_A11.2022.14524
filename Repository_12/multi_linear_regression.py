import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew
# matplotlib inline

import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = (12,8)

# load the data
df=pd.read_csv('Advertising.csv')

print(df.head(10))
print(df.info())

# Relationship between Festures and Response
sns.pairplot(df, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', height=7, aspect=0.7)
plt.show()

# estimate coefficients

from sklearn.linear_model import LinearRegression
x = df.drop('sales', axis = 1)
y = df['sales']
lm1 = LinearRegression()
lm1.fit(x,y)
print(lm1.intercept_)
print(lm1.coef_)


result = list(zip(['TV', 'radio', 'neswpaper'], lm1.coef_))
print(result)

sns.heatmap(df.corr(), annot = True)
plt.show()

# Feature Selection

from sklearn.metrics import r2_score, mean_squared_error
lm2 = LinearRegression()
lm2.fit(x[['TV','radio']],y)
lm2_pred = lm2.predict(x[['TV', 'radio']])
print(r2_score(y, lm2_pred))

lm3 = LinearRegression()
lm3.fit(x[['TV', 'radio', 'newspaper']], y)
lm3_pred = lm3.predict(x[['TV', 'radio', 'newspaper']])
print(r2_score(y, lm3_pred))

# Model Evaluation using Train and Metrics

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X = df.drop('sales', axis=1)
y = df['sales']
x_train, x_test, y_train, y_test=train_test_split(X,y, random_state=1)

lm4 = LinearRegression().fit(x_train, y_train)
lm4_pred = lm4.predict(x_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, lm4_pred)))
print("R*2: ",r2_score(y_test, lm4_pred))

from yellowbrick.regressor import PredictionError,ResidualsPlot
lm5 = LinearRegression()
v = PredictionError(lm5).fit(x_train, y_train)
v.score(x_test, y_test)
v.poof()
v.show()

# Interaction Effect
df['interaction'] = df['TV']*df['radio']
X = df[['TV', 'radio', 'interaction']]
y = df['sales']
x_train, x_test , y_train, y_test = train_test_split(X, y, random_state=1)

lm6 = LinearRegression().fit(x_train, y_train)
lm6_pred = lm6.predict(x_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, lm6_pred)))
print("R*2:", r2_score(y_test, lm6_pred))

v=PredictionError(lm6).fit(x_train,y_train)
v.score(x_test,y_test)
v.poof()
v.show()
