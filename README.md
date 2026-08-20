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

~~~
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
~~~
<img width="421" height="373" alt="image" src="https://github.com/user-attachments/assets/0487e545-e783-47b9-9230-3eab3f4acdd5" />

~~~
# ORDINAL ENCODING
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
~~~
<img width="433" height="185" alt="image" src="https://github.com/user-attachments/assets/f0247ac3-5dad-4658-8903-fe89ba75e757" />

~~~
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
~~~
<img width="596" height="374" alt="image" src="https://github.com/user-attachments/assets/1082a680-cd74-45b8-ad10-ba75dde305b6" />

~~~
 # Label Encoder ( orders in alphabetical order)
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
~~~
<img width="653" height="355" alt="image" src="https://github.com/user-attachments/assets/55bf07e3-c8a3-42a8-ac3d-789f074ada25" />

~~~
# ONE HOT ENCODING
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) # Orders in Alphabetical Order Blue , Green, Red
df2=pd.concat([df2,enc],axis=1)
df2
~~~
<img width="573" height="358" alt="image" src="https://github.com/user-attachments/assets/18ad486f-fc31-4036-8b8a-55832a9581c7" />
~~~
pd.get_dummies(df2,columns=["nom_0"])
~~~
<img width="670" height="374" alt="image" src="https://github.com/user-attachments/assets/4a05bac8-9b5a-41ac-b8ad-b20fab80b595" />

~~~
pip install --upgrade category_encoders
~~~
<img width="1046" height="275" alt="image" src="https://github.com/user-attachments/assets/440ab4ee-77a2-419a-b288-e8ff747dfc95" />

~~~
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
~~~
<img width="481" height="348" alt="image" src="https://github.com/user-attachments/assets/324f19e2-7c5d-4116-abae-b5858330297c" />

~~~
 be=BinaryEncoder()
 nd=be.fit_transform(df['Ord_2'])
 dfb=pd.concat([df,nd],axis=1)
 dfb
~~~
<img width="709" height="368" alt="image" src="https://github.com/user-attachments/assets/c4d6c0fb-712b-4579-a225-636f4967f077" />

~~~
 from category_encoders import TargetEncoder
 te=TargetEncoder()
 CC=df.copy()
 new=te.fit_transform(X=CC["City"],y=CC["Target"])
 CC=pd.concat([CC,new],axis=1)
 CC
~~~
<img width="569" height="357" alt="image" src="https://github.com/user-attachments/assets/6f57409a-58dc-4a19-bd76-5a123b867333" />

~~~
 import pandas as pd
 from scipy import stats
 import numpy as np
 df=pd.read_csv("/content/Data_to_Transform.csv")
 df
~~~
<img width="815" height="425" alt="image" src="https://github.com/user-attachments/assets/a69b6e36-7c32-4e45-833d-537dca43d592" />

~~~
df.skew()
~~~
<img width="374" height="174" alt="image" src="https://github.com/user-attachments/assets/ceed5181-6995-4fc7-a778-bd0ae52403ad" />

~~~
np.log(df["Highly Positive Skew"])
~~~
<img width="343" height="444" alt="image" src="https://github.com/user-attachments/assets/2daa4142-2b02-412b-9bef-b8ecf84b4aee" />


~~~
 np.reciprocal(df["Moderate Positive Skew"])
~~~
<img width="266" height="459" alt="image" src="https://github.com/user-attachments/assets/f90586b7-fd77-4c90-8942-05fb8a3bfe82" />

~~~
 np.sqrt(df["Highly Positive Skew"])
~~~
<img width="246" height="454" alt="image" src="https://github.com/user-attachments/assets/08be25ce-f663-4c0a-9cea-61aa7b004a0a" />

~~~
 df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
 df
~~~
<img width="984" height="462" alt="image" src="https://github.com/user-attachments/assets/3e738957-4b5f-4503-8a23-9cb9183c89c8" />

~~~
 df.skew()
~~~
<img width="386" height="215" alt="image" src="https://github.com/user-attachments/assets/fd25a0ce-6cdb-4625-99fc-342137aaca4d" />

~~~
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
~~~
<img width="371" height="265" alt="image" src="https://github.com/user-attachments/assets/ae2370ff-585a-4856-bb3b-b8b460253ab2" />

~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
~~~
<img width="997" height="465" alt="image" src="https://github.com/user-attachments/assets/299cd68a-5332-4303-9117-0f7cd8da17fe" />

~~~
import seaborn as sns
import statsmodels.api as sm 
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45') 
plt.show()
~~~
<img width="664" height="445" alt="image" src="https://github.com/user-attachments/assets/72505ad0-4b2e-4990-9ce6-a5e88ed6d87f" />

~~~
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') 
plt.show()
~~~
<img width="659" height="443" alt="image" src="https://github.com/user-attachments/assets/7050f20d-4a71-4fab-b2f6-14e71a0a1f54" />

~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~
<img width="681" height="460" alt="image" src="https://github.com/user-attachments/assets/c3d00ab6-3d6a-4797-918e-ea92bdc13080" />

# RESULT:
The program is excuted successfully
       
