#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#function to deal with outliers in a dataset
def fix_outliers(dataset,col_lst):
    for col in col_lst:
        q1=dataset[col].quantile(0.25)
        q3=dataset[col].quantile(0.75)
        IQR=q3-q1
        lower=q1-(1.5*IQR)
        upper=q3+(1.5*IQR)
        p5=dataset[col].quantile(0.05)
        p95=dataset[col].quantile(0.95)
        dataset.loc[dataset[col]<lower,col]=p5
        dataset.loc[dataset[col]>upper,col]=p95
    return dataset

#function that applies all feature engineering to dataset
def feat_eng(dataset):
    #creating the new features
    dataset['Age/Tenure']=dataset['Age']/dataset['Tenure']
    dataset['Balance/Tenure']=dataset['Balance']/dataset['Tenure']
    dataset['Balance/Age']=dataset['Balance']/dataset['Age']

    #dealing with NA/inf values
    dataset=dataset.replace([np.nan,np.inf,-np.inf], 0)

    return dataset

#function for applying preprocessing found in Preprocessing_Visualization notebook
def load_preprocess_data(file,outliers=False): #allows choice to fix outliers or not; default is deal with outliers
    df=pd.read_csv(file)
    df=df.iloc[:,3:] #dropping first 3 columns (ID columns)

    #dealing with outliers
    if outliers==False:
        need_fix=['CreditScore','Age'] #from table in Data Understanding section
        df=fix_outliers(df,need_fix)

    #feature engineering
    df=feat_eng(df)

    #one hot encoding categorical columns
    cat_varnames=['Geography','Gender','Tenure','NumOfProducts','HasCrCard','IsActiveMember']
    encoder=OneHotEncoder()
    encoder.fit(df[cat_varnames])
    encoded = pd.DataFrame(encoder.transform(df[cat_varnames]).toarray())

    #normalizing numerical columns
    nums=df.drop(columns=cat_varnames)
    nums=nums.drop(columns='Exited') #also dropping target var
    scaler=MinMaxScaler()
    nums=pd.DataFrame(scaler.fit_transform(nums),
                      columns=['CreditScore','Age','Balance','EstimatedSalary','Age/Tenure','Balance/Tenure','Balance/Age'])

    #recombining numerical and categorical 
    df=pd.concat([nums,encoded,df['Exited']],axis=1)

    return df

def testdf_prep(file,outliers=False):
    df=pd.read_csv(file)
    df=df.iloc[:,3:] #dropping first 3 columns (ID columns)

    #dealing with outliers
    if outliers==False:
        need_fix=['CreditScore','Age'] #from table in Data Understanding section
        df=fix_outliers(df,need_fix)

    #feature engineering
    df=feat_eng(df)

    #one hot encoding categorical columns
    cat_varnames=['Geography','Gender','Tenure','NumOfProducts','HasCrCard','IsActiveMember']
    encoder=OneHotEncoder()
    encoder.fit(df[cat_varnames])
    encoded = pd.DataFrame(encoder.transform(df[cat_varnames]).toarray())

    #normalizing numerical columns
    nums=df.drop(columns=cat_varnames)
    scaler=MinMaxScaler()
    nums=pd.DataFrame(scaler.fit_transform(nums),
                      columns=['CreditScore','Age','Balance','EstimatedSalary','Age/Tenure','Balance/Tenure','Balance/Age'])

    #recombining numerical and categorical 
    df=pd.concat([nums,encoded],axis=1)

    return df