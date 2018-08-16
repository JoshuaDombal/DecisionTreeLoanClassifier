import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', 1000)

loans = pd.read_csv('loan_data.csv')
print(loans.head())
print(loans.info())
print(loans.describe())

#loans.hist(column='credit.policy')
#loans.hist(column='not.fully.paid')

#sns.lmplot(x='not.fully.paid',y='log.annual.inc',data = loans)
#sns.pairplot(loans)

#sns.distplot(loans['not.fully.paid'])

'''
loans[loans['credit.policy']==1]['fico'].hist(bins=35, color='blue',
                                              label='Credit Policy = 1',
                                              alpha=.6)
loans[loans['credit.policy']==0]['fico'].hist(bins=35, color='red',
                                              label='Credit Policy = 0',
                                              alpha=.6)
plt.show()

loans[loans['not.fully.paid']==1]['fico'].hist(bins=35, color='blue',
                                              label='Not fully Paid = 1',
                                              alpha=.6)
loans[loans['not.fully.paid']==0]['fico'].hist(bins=35, color='red',
                                              label='Not Fully Paid = 0',
                                              alpha=.6)

plt.show()
'''
#plt.figure(figsize=(11,7))
#sns.countplot(x='purpose',hue='not.fully.paid',data=loans, palette='Set1')
#plt.show()


#sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
#plt.show()

plt.figure(figsize=(11,7))
sns.lmplot(x='fico',y='int.rate',data=loans,hue='credit.policy',col='not.fully.paid',palette='Set1')
#plt.show()

# Get rid of categorical feature

cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
print(final_data.head())

from sklearn.model_selection import train_test_split

X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))


############################### RANDOM FOREST MODEL ###########################################

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)

predictions2 = rfc.predict(X_test)
print(classification_report(y_test, predictions2))
print(confusion_matrix(y_test, predictions2))
