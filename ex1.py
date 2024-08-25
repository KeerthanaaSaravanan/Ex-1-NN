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