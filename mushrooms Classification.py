#Import the libraries
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

#Read the file
data = pd.read_csv("mushrooms.csv")

#print first 10 rows, to see if the file is properly loaded
print(data.head(10))

#show the sum of all null values in all columns individually
print(data.isnull().sum())

#Check how many classes are there, that we need to predict
print(data['class'].unique())

#Convert all alphabets into integers, we use label encoder for that
labelencoder = LabelEncoder()

#Select all columns one by one and "fit" the label encoder on them
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])
    
#Print the first 10 encoded values
print(data.head(10))

# all rows, all the features and no labels
X = data.iloc[:,1:23]

# all rows, label only
y = data.iloc[:, 0]

#Split the data into training and testing sets (80-20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=142)

clf = MLPClassifier(hidden_layer_sizes=(6), max_iter=5000, alpha=0.0001, solver='sgd', random_state=145, verbose=10, tol=0.000000001)
clf.fit(X_train, y_train)

Y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test,Y_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (fn+tp)
specificity = fp / (fp+tn)
precision = tp / (tp+fp)
f1score = 2 * ((precision*sensitivity)/(precision+sensitivity))

print(cm)
sns.heatmap(cm)
print("\nSensitivity:")
print(sensitivity)
print("\nSpecificity:")
print(specificity)
print("\nAccuracy:")
print(accuracy_score(y_test,Y_pred))
print("\nPrecision:")
print(precision)
print("\nF1 Score:")
print(f1score)