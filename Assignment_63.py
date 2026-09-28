import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


df = pd.read_csv("Loan_Default.CSV")

print("first few data of dataset is :")
print(df.head())

# step 2 : EDA

print("check for missing values in dataset :")
print(df.isnull().sum())


# step 3 : check wether the target class is balanced or not

print("check wether the target class is balanced or not :")
print(df["Default"].value_counts())

# step 4 : endod categorical columns into numerical columns using one hot encoding

df = pd.get_dummies(df, columns=["PreviousDefault","HomeOwnership"],drop_first=True, dtype=int)

print("first few data of dataset after one hot encoding is :")
print(df.head())

# step 5 : seprate x and y

x = df.drop("Default", axis=1)
y = df["Default"]

print("independent variables are :")
print(x.head())

print("dependent variable is :")   
print(y.head())

# step 6 : split the dataset into traing and testting datase

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)

print("shape of x_train is :", x_train.shape)
print("shape of x_test is :", x_test.shape)
print("shape of y_train is :", y_train.shape)
print("shape of y_test is :", y_test.shape)

# step 7 : explain wether startified split is used or not

print("explain wether startified split is used or not :")
print("Yes, stratified split is used to ensure that the proportion of each class is maintained in both training and testing sets.")

# step 8 : scale the features

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# step 9 : create and train the MLP model

model = MLPClassifier(

    hidden_layer_sizes=(32,16),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42

)

# step 10 : train the model

model = model.fit(x_train, y_train)

# step 11 : calculate the accuracy of the model

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model is :", accuracy)

# step 12 : confusion matrix

cm = confusion_matrix(y_test, y_pred)
print("confussion matrix is :", cm)

# step 13 : geerate classification report

cr = classification_report(y_test, y_pred)
print("classification report is :")
print(cr)

# step 14 : calculate the precision, recall and f1-score

precision = cm[1][1] / (cm[1][1] + cm[0][1])
recall = cm[1][1] / (cm[1][1] + cm[1][0])
f1_score = 2 * (precision * recall) / (precision + recall)

print("Precision is :", precision)
print("Recall is :", recall)
print("F1-Score is :", f1_score)

# step 15 : plot training loss 

plt.plot(model.loss_curve_)
plt.title("Training Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

# step 16 : Test the model with a new data point

new_data = [[30000, 1, 1, 0, 1]]  # Example new data point
new_data = scaler.transform(new_data)
prediction = model.predict(new_data)
print("Prediction for new data point is :", prediction)


