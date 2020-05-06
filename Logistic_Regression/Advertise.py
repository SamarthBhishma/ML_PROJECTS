import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix


# load the dataset using pandas
advertise=pd.read_csv("advertising.csv")

# get column names from dataset
print(advertise.columns.values)

# get staticial description of data
print(advertise.describe())

# create a histogram for age inorder to know the age group involved
plt.hist(advertise['Age'],bins=44)
plt.show()
# analyse the income of age group using join plot
sns.jointplot(x='Age',y='Area Income',data=advertise)
plt.show()

# analyse daily time spent on site by age groups
sns.jointplot(x="Age",y='Daily Internet Usage',data=advertise,kind='hex')
plt.show()

X=advertise[['Daily Time Spent on Site' ,'Age' ,'Area Income', 'Daily Internet Usage','Male']]
y=advertise['Clicked on Ad']

print(X.shape)
print(y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)

prediction=logmodel.predict(X_test)
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))



