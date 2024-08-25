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
### Missing Data
![Screenshot 2024-08-23 213007](https://github.com/user-attachments/assets/5a748f87-8d73-4ab5-a832-b4fa59600829)



### Number of duplicates
![2](https://github.com/user-attachments/assets/b7ea83e0-3c68-43f9-8499-97af0babb72a)

<br>

### Outliers
![3](https://github.com/user-attachments/assets/1602f7d1-5dd0-4ca7-9c2c-d6d352d02f13)

<BR>
<BR>

### Normalized dataset
![4](https://github.com/user-attachments/assets/f57415b3-f3e2-421b-94e6-bd4612cba991)


<BR>

### X and Y
![5](https://github.com/user-attachments/assets/5198e027-e8d3-4b98-91ea-f8e1a6df8f95)


<BR>
<BR>
  
### X_train,X_test,Y_train,Y_test
![6](https://github.com/user-attachments/assets/76a8bfa1-a625-4167-a220-2403ebe77af3)


<BR>

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


