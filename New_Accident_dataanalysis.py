# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 16:16:49 2025

@author: Hi
"""

#importing all required libraries
import pandas as pd       #importing pandas for data manipulaion
import numpy as np        #importing numpy for numerical computations
import matplotlib.pyplot as plt#importing matplotlibrary for visualization
import seaborn as sns # For enhanced plotting and visualization
import warnings                    #importing warnings
warnings.filterwarnings('ignore')       #ignoring the warnings
from feature_engine.outliers import Winsorizer #importing winsorizer for outliers treatment
from sklearn.impute import SimpleImputer#importing simpleimputer from sklearn
from sklearn.preprocessing import MinMaxScaler,FunctionTransformer#importing minmax scalar for scaling
from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures# importing onehot encoder for encoding

data1=pd.read_csv(r"C:\Users\admin\Downloads\Project_dataset (1).csv")
data1.info()

#EDA----------------------------------------------
#calculating descriptive statistics
d=data1.describe()
#-------firt moment of busiess decision
num=data1.select_dtypes(include=['number'])
cat=data1.select_dtypes(include=['object'])

#calculating mean of numerical columns
rows=[]
for i in num:
     m=data1[i].median()
     d=data1[i].mean()
     s=data1[i].std()
     v=data1[i].var()
     skew=data1[i].skew()
     k=data1[i].kurt()
     rows.append({'column':i,'mean':d,'median':m,'stand_dev':s,'variance':v,'skewess':skew,'kurtosis':k})
des_stats=pd.DataFrame(rows)
print(des_stats)

#calculating mode of categorical columns
cat['road_type'].value_counts()
cat['lighting'].value_counts()
cat['weather'].value_counts()
cat['enforcement_level'].value_counts()
cat['traffic_data_quality_flag'].value_counts()
cat['signal_status'].value_counts()
cat['signal_data_quality_flag'].value_counts()

#checking duplicates
data1.duplicated().sum()
#checking missing values
data1.isnull().sum()

#checking outliers values
d=[]
for i in num:
    iqr=data1[i].quantile(0.75)-data1[i].quantile(0.25)
    minimum=data1[i].quantile(0.25)-(iqr*1.5)
    maximum=data1[i].quantile(0.75)+(iqr*1.5)
    outliers=np.where(data1[i]>maximum, True,np.where(data1[i]<minimum,True,False))
    k=outliers.sum()
    d.append([i,k])

#data visualisations
#Histogram
plt.hist(data1['lane_count'])
plt.hist(data1['blackspot_score'])
plt.hist(data1['vehicle_count_per_hr'])
plt.hist(data1['speed_limit_kmph'])
plt.hist(data1['avg_speed_kmph'])
plt.hist(data1['green_duration_s'])
plt.hist(data1['yellow_duration_s'])
plt.hist(data1['red_duration_s'])
plt.hist(data1['cycle_time_s'])
plt.hist(data1['violations_count'])
plt.hist(data1['veh_count_at_accident'])

#boxplot for checking outliers
plt.boxplot(data1['speed_limit_kmph'],vert=False)
plt.boxplot(data1['blackspot_score'],vert=False)
plt.boxplot(data1['vehicle_count_per_hr'],vert=False)
plt.boxplot(data1['avg_speed_kmph'],vert=False)
plt.boxplot(data1['green_duration_s'],vert=False)
plt.boxplot(data1['red_duration_s'],vert=False)
plt.boxplot(data1['yellow_duration_s'],vert=False)
plt.boxplot(data1['cycle_time_s'],vert=False)
plt.boxplot(data1['veh_count_at_accident'],vert=False)


#heatmap
corr=num.corr()
sns.heatmap(corr)

#------data preprocessing---------------
#select the required columns

data=data1.drop(['location_id','timestamp','state','day_of_week','lighting','hour_of_day','latitude','longitude','season','peak'],axis=1)


df1=data1[[
  "vehicle_count_per_hr",
  "blackspot_score",
  "weather",
  "is_peak",
  "enforcement_level",
  "lane_count",
  "speed_limit_kmph",
  "road_type",
  'has_signal',
  'violations_count',
  'green_duration_s',
  'red_duration_s'
]]


df1.select_dtypes(include=['number']).columns
#converting enforcement level into numerical format
df1['enforcement_level'] = df1['enforcement_level'].map({'low':0,'medium':1,'high':2}).astype("int64")
df1['is_peak']=df1['is_peak'].astype('str')
df1['has_signal']=df1['has_signal'].astype('str')

#-----------Missing value treatment
df=df1.copy()
#missing values treatment on categorical columns
impute=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
df['weather']=pd.DataFrame(impute.fit_transform(df[['weather']]))
df['vehicle_count_per_hr']=pd.DataFrame(impute.fit_transform(df[['vehicle_count_per_hr']]))
df['violations_count']=pd.DataFrame(impute.fit_transform(df[['violations_count']]))

#missing values treatment on numerical columns
impute2=SimpleImputer(missing_values=np.nan,strategy='median')

df['green_duration_s']=pd.DataFrame(impute2.fit_transform(df[['green_duration_s']]))
df['red_duration_s']=pd.DataFrame(impute2.fit_transform(df[['red_duration_s']]))


#-----------Outliers treatment------------------
#importing winsorizer for outliers treatment
from feature_engine.outliers import Winsorizer

Outliers=Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=['speed_limit_kmph','blackspot_score'])

df2=pd.DataFrame(Outliers.fit_transform(df[['speed_limit_kmph','blackspot_score']]))


Outliers1=Winsorizer(capping_method='mad',
                    tail='both',
                    fold=1.5,
                    variables=['green_duration_s','red_duration_s'])

df3=pd.DataFrame(Outliers1.fit_transform(df[['red_duration_s','green_duration_s']]))

Outliers2=Winsorizer(capping_method='mad',
                    tail='both',
                    fold=3,
                    variables=['vehicle_count_per_hr'])

df4=pd.DataFrame(Outliers2.fit_transform(df[['vehicle_count_per_hr']]))

#cheking outliers removed or not
plt.boxplot(df2['speed_limit_kmph'],vert=False)
plt.boxplot(df2['blackspot_score'],vert=False)
plt.boxplot(df3['red_duration_s'],vert=False)
plt.boxplot(df3['green_duration_s'],vert=False)
plt.boxplot(df4['vehicle_count_per_hr'],vert=False)


data_1=pd.concat([df2,df3,df4,df[['lane_count','violations_count','enforcement_level']]],axis=1)
#scaling
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
data_2=pd.DataFrame(scale.fit_transform(data_1),columns=data_1.columns)

#select category columns
cat_columns=df.select_dtypes(include=['object'])

#encoding categorical variables using onehot encoder
encode=OneHotEncoder(sparse_output=False,drop='first')
encoded=pd.DataFrame(encode.fit_transform(cat_columns),columns=encode.get_feature_names_out(input_features=cat_columns.columns))

clean_data=pd.concat([data_2,encoded],axis=1)


# Combine original and encoded data
clean_data = pd.concat([data_2, encoded], axis=1).reset_index(drop=True)

#  POLYNOMIAL FEATURES (DEGREE = 2)
poly = PolynomialFeatures(degree=2, include_bias=False)

# Fit and transform only the scaled numeric columns
poly_features = poly.fit_transform(clean_data[data_2])

# Get new column names
poly_feature_names = poly.get_feature_names_out(data_2)

# Create DataFrame for polynomial features
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)

# Combine original clean_data with polynomial features
df_final = pd.concat([clean_data.reset_index(drop=True), df_poly], axis=1)

# Save final dataset
df_final.to_csv("Accident_data.csv", index=False)

# Display output
print("\nFINAL MODEL-READY DATA (With Polynomial Features):")
print(df_final.head())

print("\nAll processing steps completed successfully!")
