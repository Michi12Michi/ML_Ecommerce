import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv("Ecommerce Customers")
df.describe()
df.info()

sns.jointplot(data=df, x="Time on Website", y="Yearly Amount Spent")
sns.jointplot(data=df, x="Time on Website", y="Time on App")
sns.jointplot(data=df, x="Time on App", y="Length of Membership", kind="hex")
sns.pairplot(data=df)
# it seems that a correlation exists among both Time on App and Yearly amount spent and Length of Membership and Yearly amount spent.
sns.heatmap(df.corr(numeric_only=True), annot=True)
# the heatmap briefly confirms.

sns.lmplot(x="Length of Membership", y="Yearly Amount Spent", data=df)

X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
Y = df['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.coef_)
pd.DataFrame(lm.coef_, X.columns, columns=["Coefficient"])

preds = lm.predict(X_test)
sns.scatterplot(x=y_test, y=preds)

from sklearn import metrics
print("MAE: ", metrics.mean_absolute_error(y_test, preds))
print("MSE: ", metrics.mean_squared_error(y_test, preds))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, preds)))
print("R**2: ", metrics.explained_variance_score(y_test, preds))
# 0.99

sns.displot(y_test-preds, kind="kde")
