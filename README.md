## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```python
      import pandas as pd
df = pd.read_csv("data.csv")
df
```
<img width="512" height="374" alt="image" src="https://github.com/user-attachments/assets/732a7c6a-d71a-4775-a865-3a1efee16e2e" />

```python
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
order1 = ['High School','Diploma','Bachelors','Masters','PhD']
enc = OrdinalEncoder(categories=[order1])
df['ordenc1']=enc.fit_transform(df[['Ord_2']])
order2=['Cold','Warm','Hot','Very Hot']
enc1 = OrdinalEncoder(categories=[order2])
df['ordenc2']=enc1.fit_transform(df[['Ord_1']])
df
```
<img width="736" height="420" alt="image" src="https://github.com/user-attachments/assets/dd7d41df-2d62-456d-9b15-26e26d940fb9" />

```python
enc3 = LabelEncoder()
df['LabelEncoder']=enc3.fit_transform(df[['Ord_2']])
df
```
<img width="840" height="411" alt="image" src="https://github.com/user-attachments/assets/08eec69f-0183-46bf-bd52-2f986a95b0d4" />

```python
from sklearn.preprocessing import OneHotEncoder
enc2 = OneHotEncoder()
newdata = pd.DataFrame(enc2.fit_transform(df[['Ord_2']]))
df2 = pd.concat([df,newdata],axis=1)
df2
```
<img width="1113" height="382" alt="image" src="https://github.com/user-attachments/assets/c4a9b4eb-c4d8-49b1-9305-cd44843bad35" />

```python
df3=pd.get_dummies(df2,columns=['City'])
df3
```
<img width="1418" height="359" alt="image" src="https://github.com/user-attachments/assets/e3f02b00-31fd-40fe-a405-11ed1e406b7f" />

```python
from category_encoders import TargetEncoder
enc5 = TargetEncoder()
df6 = df2.copy()
newdata = pd.DataFrame(enc5.fit_transform(df5[['Ord_1']],df5[['Target']]))
df7 = pd.concat([df6,newdata],axis=1)
df7
```
<img width="1138" height="373" alt="image" src="https://github.com/user-attachments/assets/bbab132f-3f2a-4bb8-9154-633815ac8a21" />

<img width="387" height="56" alt="image" src="https://github.com/user-attachments/assets/0989f5b3-3488-4a24-97f4-185e1db749f8" />

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#stats for boxcox and yeo johnson methods
import statsmodels.api as sm
df = pd.read_csv("Data_to_Transform.csv")
df
```
<img width="770" height="404" alt="image" src="https://github.com/user-attachments/assets/3703bbcf-941f-48ff-9416-520d11eead5a" />

```python
df['boxcox1'],parameters=stats.boxcox(df['Highly Positive Skew'])
sm.qqplot(df['boxcox1'],line="45")
plt.show()
```
<img width="601" height="448" alt="image" src="https://github.com/user-attachments/assets/3deebd6b-18f5-4fed-b4aa-7c271c783874" />

```python
df['boxcox2'],parameters=stats.boxcox(df['Moderate Positive Skew'])
sm.qqplot(df['boxcox2'],line="45")
plt.show()
```
<img width="615" height="447" alt="image" src="https://github.com/user-attachments/assets/dc70a273-6ad8-4258-85b6-3224777574cd" />

```python
df['yeojohnson1'],parameters=stats.yeojohnson(df['Highly Negative Skew'])
sm.qqplot(df['yeojohnson1'],line="45")
plt.show()
```
<img width="598" height="450" alt="image" src="https://github.com/user-attachments/assets/3546ae7a-f4eb-4f34-abd0-07f9d0f3d128" />

```python
df['yeojohnson2'],parameters=stats.yeojohnson(df['Moderate Negative Skew'])
sm.qqplot(df['yeojohnson2'],line="45")
plt.show()
```
<img width="599" height="453" alt="image" src="https://github.com/user-attachments/assets/6c1b59f3-7b36-406f-998d-67453c81dc6a" />

```python
sm.qqplot(np.log(df['Highly Positive Skew']),line="45")
plt.show()
```
<img width="591" height="461" alt="image" src="https://github.com/user-attachments/assets/adbebf44-d359-4e85-948f-4602075f9df5" />

```python
sm.qqplot((df['Highly Negative Skew'])**2,line="45")
plt.show()
```
<img width="602" height="460" alt="image" src="https://github.com/user-attachments/assets/dc18ac77-0a74-47f0-9955-d974867aa4cd" />

```python
sm.qqplot(np.log(df['Moderate Positive Skew']),line="45")
plt.show()
```
<img width="608" height="449" alt="image" src="https://github.com/user-attachments/assets/27d0e00c-9288-4fa5-94aa-03d200089f28" />

```python
sm.qqplot((df['Moderate Negative Skew'])**2,line="45")
plt.show()
```
<img width="590" height="456" alt="image" src="https://github.com/user-attachments/assets/492315c0-6209-4f0a-b72a-261c7feb2f8a" />

```python
from sklearn.preprocessing import QuantileTransformer
enc = QuantileTransformer(output_distribution="normal")
df['quantile'] = enc.fit_transform(df[['Highly Positive Skew']]).flatten()
df
```
<img width="806" height="443" alt="image" src="https://github.com/user-attachments/assets/80b3e081-9a84-4ea6-8342-00e44be2d13f" />

```python
sm.qqplot(df['quantile'],line="45")
plt.show()
```
<img width="583" height="452" alt="image" src="https://github.com/user-attachments/assets/242b4e70-2c17-4e96-b574-f6495d410de0" />

```python
enc = QuantileTransformer(output_distribution="normal")
df['quantile2'] = enc.fit_transform(df[['Highly Negative Skew']]).flatten()
sm.qqplot(df['quantile2'],line="45")
plt.show()
```
<img width="586" height="450" alt="image" src="https://github.com/user-attachments/assets/7603a52b-0559-4fb3-b7fb-ce75d0851f85" />

```python
enc = QuantileTransformer(output_distribution="normal")
df['quantile3'] = enc.fit_transform(df[['Moderate Negative Skew']]).flatten()
sm.qqplot(df['quantile3'],line="45")
plt.show()
```
<img width="589" height="443" alt="image" src="https://github.com/user-attachments/assets/9cb9654d-5dcc-4897-b7d7-ea8e0b8ffa7a" />

```python
enc = QuantileTransformer(output_distribution="normal")
df['quantile4'] = enc.fit_transform(df[['Moderate Positive Skew']]).flatten()
sm.qqplot(df['quantile4'],line="45")
plt.show()
```
<img width="591" height="446" alt="image" src="https://github.com/user-attachments/assets/300ec00e-5d3f-45b3-a1fa-2f41bc9d988e" />























# RESULT:
The given data has been read, Feature Encoding and Transformation process have been performed and the data has been saved to a file.

       
