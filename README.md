<br>
<H3>NAME: Keerthana S</H3>
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
#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset from drive
df=pd.read_csv('/content/Churn_Modelling.csv')
df

# Finding Missing Values
print(df.isnull().sum())

#Handling Missing values
df.fillna(df.mean(),inplace=True)
print(df.isnull().sum())

y=df.iloc[:,-1].values
print(y)

#Check for Duplicates
df.duplicated()

#Detect Outliers
df.describe()

#Normalize the dataset
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

#split the dataset into input and output
x=df.iloc[:, :-1].values
print(x)
y=df.iloc[:,-1].values
print(y)

#splitting the data for training & Testing
X_train ,X_test ,y_train,y_test=train_test_split(x,y,test_size=0.2)

#Print the training data and testing data
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))

```
## OUTPUT:
### Dataset:
![image](https://github.com/user-attachments/assets/b63534a2-dfd2-45c1-b75f-ea3144cdf36b)



### Finding Missing Values:
![image](https://github.com/user-attachments/assets/2bc4209a-27ed-4ca2-bf7f-03c7c1c75e7b)


<br>

### Handling Missing values:
![image](https://github.com/user-attachments/assets/051705cf-b0a4-4449-89e9-992295bf90d9)


<BR>
<BR>

### Duplicates:
![image](https://github.com/user-attachments/assets/fbb42d1c-5d75-4bbc-871e-03490d2b59c2)



<BR>

### Normalized dataset:
![image](https://github.com/user-attachments/assets/c1ca252f-ddf4-441d-a7a8-8b8302278591)



<BR>
<BR>
  
### Split the dataset into input and output:
![image](https://github.com/user-attachments/assets/f70a525d-0932-4be4-bc44-c63d2f975d1f)



### Splitting the data for training & Testing:
![image](https://github.com/user-attachments/assets/d7a99a72-7faf-4f84-89a5-9d9c091a78b4)


<BR>

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


