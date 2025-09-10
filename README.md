# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary. 6.Define a function to predict the Regression value. .
## Program:
**Developed by: SHARON CLARA A**

**RegisterNumber:  212224040310**
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the file
dataset=pd.read_csv("Placement_Data.csv")
dataset

#dropping the serial no and salary col
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

#categorizing col for further labelling
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

#labelling the columns
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

#display dataset
dataset

#selecting the features and labels
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values

#display dependent variables
Y

#Inititalize the model parameters
theta=np.random.randn(X.shape[1])
y=Y
#Define the sigmoid function
def sigmoid(z):
    return 1 /(1 + np.exp(-z))

#Define the loss function
def loss(theta,X,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

#Define the gradient descent algorithm

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha * gradient
    return theta

#Train the model
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

#Make predictions
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred

y_pred=predict(theta,X)

#Exaluate the model
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:
**Dataset**

<img width="1482" height="507" alt="486492554-337943b0-7362-4b9a-9d03-09d2f90e3c80" src="https://github.com/user-attachments/assets/5400df15-ad3d-49bb-80dc-001929daf2dc" />

**Dataset.dtypes**

<img width="371" height="386" alt="486492649-5779d171-92ad-4ca7-b5ed-c98d95bfdd4a" src="https://github.com/user-attachments/assets/0704e4a9-87be-4949-be4e-6e11c275e647" />

**Dataset**

<img width="1175" height="562" alt="486492705-44fe9ae6-303e-4a93-905d-bd291c0d893c" src="https://github.com/user-attachments/assets/9361096b-2f31-4586-b0c1-8c5659d8e998" />

**Y**

<img width="891" height="283" alt="486492751-f5632cd0-2c42-46cf-818c-0fd3abea38bc" src="https://github.com/user-attachments/assets/4837c8c6-818d-4960-a96a-c0ed42f59a3c" />
<img width="422" height="56" alt="486492828-6fbf1384-e748-463f-956d-ecf5f0d115d9" src="https://github.com/user-attachments/assets/7d2e35e6-1f1a-400f-b4c9-a1e823352ac6" />

**Y_Pred**

<img width="912" height="180" alt="486492895-b6e49210-b49a-4287-a96a-10b0752b2303" src="https://github.com/user-attachments/assets/3c8144a9-4db8-40c2-b819-5935f272e8ed" />

**Y**

<img width="921" height="178" alt="486492930-f6078e62-20c8-44ea-bfc2-4e72c760094f" src="https://github.com/user-attachments/assets/006311ed-4298-4c6e-a1ea-deec08369c0a" />

**Y_prednew**

<img width="173" height="42" alt="486492974-63edad46-4bbf-45e2-9f21-bf1a67f84b25" src="https://github.com/user-attachments/assets/c46d716f-e3ef-4aeb-a962-e8318f8ef708" />

**Y_prednew**

<img width="111" height="51" alt="486493000-3e40614a-76cc-4e53-a4f4-1ad28cc75b7d" src="https://github.com/user-attachments/assets/d21c6f67-13d7-4da9-95bb-3aae6197ce2a" />







## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

