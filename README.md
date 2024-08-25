<br>
<H3>NAME: KEERTHANA S</H3>
<H3>REGISTER NO: 212223240070</H3>
<H3>EX. NO.1</H3>
<H3>DATE:22-08-2024</H3>
<br>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>
<br>

## AIM:
To perform Data preprocessing in a data set downloaded from Kaggle.

<br>

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

<br>

## RELATED THEORETICAL CONCEPT:
<br>

**Kaggle :**

Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

<br>

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

<br>

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.

<br>
<br>
<br>

## ALGORITHM:
<br>

#### STEP 1:
Importing the libraries<BR>
#### STEP 2:
Importing the dataset<BR>
#### STEP 3:
Taking care of missing data<BR>
#### STEP 4:
Encoding categorical data<BR>
#### STEP 5:
Normalizing the data<BR>
#### STEP 6:
Splitting the data into test and train<BR>

<br>

##  PROGRAM:
```
#Import Libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset
df=pd.read_csv("/content/Churn_Modelling.csv")

#Check the missing data
df.isnull().sum()

# Finding Missing Values
print(df.isnull().sum())

#Check for Duplicates
df.duplicated()

#Assigning Y
y = df.iloc[:, -1].values
print(y)

#Check for duplicates
df.duplicated()

#Check for outliers
df.describe()
print(df.describe())

#Dropping string values data from dataset
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

#Normalize the dataset
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

#Split the dataset
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)

#Splitting the data for training & testing
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

#Training and testing model
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```
## OUTPUT:
### Missing Values:
![1](https://github.com/user-attachments/assets/3af96bcb-8bc0-4407-ba2c-5c19224b7d1e)

### Outliers:
![2](https://github.com/user-attachments/assets/95e926f6-d411-482f-99e7-d6a0192c4111)

### Normalized dataset:
![3](https://github.com/user-attachments/assets/d4fff95e-b747-4477-8e90-92a230ff93fb)

  
### Input & Output Values:
![4](https://github.com/user-attachments/assets/64d0988e-3d02-458b-806d-e0e6bdb29c70)

### Splitting the data for training & Testing:
![5](https://github.com/user-attachments/assets/e8efc242-91f5-4268-9392-0610103a1ca6)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


