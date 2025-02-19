# consignment-pricing-predection
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.pandas.set_option('display.max_columns',None) 
('max_rows', 90)
        data = pd.read_csv('../input/supply-chain-shipment-pricing-data/SCMS_Delivery_History_Dataset.csv')
data.head(5)
data.shape
data.info()
data.info()
data1 = data.copy()

def missing_values():
    for column in data1.columns:
        if data1[column].isnull().sum()>0:
            print(f"{column} has {data1[column].isnull().sum()} missing values")

            
missing_values()
data2 = data1.copy()

data2.select_dtypes(np.number).columns
data3 = data2.copy()

for column in data3.select_dtypes(np.number).columns:
    sns.histplot(data3[column], kde=True)
    plt.show()
   
   data4 = data3.copy()
data4.select_dtypes('object').columns
for column in data4.select_dtypes('object').columns:
    print(f"{column} has {len(data4['Project Code'].unique())} unique categories")
    data5 = data4.copy()


for feature in data5.select_dtypes(np.number).columns:
    
    data5.boxplot(column=feature)
    plt.ylabel(feature)
    plt.title(feature)
    plt.show()
