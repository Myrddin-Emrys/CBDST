#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# In[2]: Getting the data
data = pd.DataFrame(pd.read_excel("socdem2.xlsx", sheet_name=0, index_col=None))


# In[3]: Checking the header
data.columns


# In[4]: Dropping the columns that do not refer to respondents soc.-dem. characteristics, creating clusterization dataset
# ('get_id_2')
get_id = data.drop(
    columns=['bdate', 'shop_type_name', 'network_name', 'shop_name', 'top_category', 'category', 'brand_name', 'price',
             'total_weight', 'quantity', 'barcode_value', 'items_per_pack',
             'purchase_id', 'promo'])


# In[5]: Making the same age for the respondents in 2020 data as in 2019 data
get_id.loc[get_id['period'] == '2020-03', 'age'] = get_id['age'] - 1
get_id.loc[get_id['period'] == '2020-04', 'age'] = get_id['age'] - 1


# In[6]: Excluding date, and IDs varaibles from clusterization dataset
get_id_2 = get_id.drop(columns=['period', 'user_id', 'person_id'])
get_id_2.columns


# Getting unique values for each variable
for i in range(len(get_id_2.columns)):
    print(get_id_2.columns[i], get_id_2[get_id_2.columns[i]].unique())


# In[7]:
get_id_2[get_id_2.columns[0]].unique()


# In[8]: Testing FIRST algorithm. The main idea is to get the ID by simply aggregating all the values (of soc.-dem.
# characteristics) by the row in a string variable.
# To do this all categorial values were converted to numbers (numerated in oreder of their appearance in the dataset).
for i in range(len(get_id_2.columns)):
    for j in range(len(get_id_2[get_id_2.columns[i]].unique())):
        get_id_2[get_id_2.columns[i]] = get_id_2[get_id_2.columns[i]].replace(
            str((get_id_2[get_id_2.columns[i]].unique()[j])), str(j))


# In[9]: N/A were replaced by '99'
get_id_2 = get_id_2.fillna(99)
get_id_2


# In[10]: The result is an ID for each row, consisting of string value which is composed from all the numerical values
# from categorical data, representing combination of soc.-dem characteristics
get_id_2['age'] = data['age']
get_id_2['ID'] = data['age']
for row in range(len(get_id_2)):
    get_id_2['ID'][row] = ''.join(str(n) for n in get_id_2.iloc[row, :-1].squeeze(axis=0))


# In[11]: Testing the SECOND algorithm. It's OPTICS clusterization algorithm based on KNN.
from sklearn.cluster import OPTICS


# In[12]: As OPTICS works with categorical data, all the values (recorded as numbers, but then converted to strings
# for previous step) were converted to numeric values.
for column in range(len(get_id_2.columns) - 1):
    get_id_2.iloc[:, column] = pd.to_numeric(get_id_2.iloc[:, column])


# In[13]: most of the parameters were set as default (n_jobs = -1 for maximum performance)
clustering_2 = OPTICS(min_samples=2, n_jobs=-1).fit(get_id_2.iloc[:, :-1])
clustering_2.labels_


# In[14]: Getting IDs from OPTICS
clusters = pd.DataFrame(clustering_2.labels_)


# In[15]: Testing the THIRD algorithm. It's the sorting function, which goes through each column,
# sorting them by values and labeling each sorted group.
# As a results we get the labelled groups of all possible sorting combinations by each column.
# Which is believed to be human-like way to identify unique combinations of soc.-dem. characteristics of each individaual.
def sorting(df, cluster_id, *column):
    df['cluster'] = 0
    try:
        for i in column:
            if i != column[-1]:
                for uniq in df[i].unique():
                    cluster_id += 1
                    for iden in df['cluster'][(df.groupby(by=i).get_group(name=uniq)).index].unique():
                        df['cluster'][(df.groupby(by=i).get_group(name=uniq)).index] += cluster_id

            else:
                for uniq in df[i].unique():
                    cluster_id += 1
                    for iden in df['cluster'][(df.groupby(by=i).get_group(name=uniq)).index].unique():
                        df['cluster'][(df.groupby(by=i).get_group(name=uniq)).index] += cluster_id
                print("Sorted by: ", column[:column.index(i) + 1])
                return df
    except KeyError:
        print("Sorted by: ", column[:column.index(i) + 1])
        return df


test_3 = get_id_2.drop(columns=['ID'])
sorting(test_3, 0, 'age', 'city', 'income', 'education', 'family', 'occupation', 'prosperity', 'child', 'hh_size',
        'animals', 'position')


# In[16]: Getting the ID labels from original data ('person_id') and three algorithms to estimate the accuracy
clustering_score = pd.DataFrame(data['person_id'])
clustering_score['ID_1_str'] = get_id_2['ID']
clustering_score['ID_2_OPTICS'] = clustering_2.labels_
clustering_score['ID_3_filter'] = test_3['cluster']
clustering_score.head()


# In[17]: Coverting string values got from first algorithm to float type in order to estimate accuracy
clustering_score['ID_1_str'] = clustering_score['ID_1_str'].replace('-1', '99', regex=True)
clustering_score['ID_1_str'] = clustering_score['ID_1_str'].astype(float)


# In[18]: In order to compare accuracy dataset of obtained labels was sorted by
# values of target (need to be predicted) variable.
# For each group the median values of labels returned by each algorithm were calculated.
clustering_score['ID_1_acc'] = clustering_score.groupby("person_id")['ID_1_str'].transform('median')
clustering_score['ID_2_acc'] = clustering_score.groupby("person_id")['ID_2_OPTICS'].transform('median')
clustering_score['ID_3_acc'] = clustering_score.groupby("person_id")['ID_3_filter'].transform('median')
clustering_score.head()


# In[19]: Accuracy were estimated by the rate of correspondence of median label of each algorithm for each target
# variable group (rows referred to unique individual) and labels returned by algorithms
ID_1_acc = (clustering_score['ID_1_str'] == clustering_score['ID_1_acc']).mean()
ID_2_acc = (clustering_score['ID_2_OPTICS'] == clustering_score['ID_2_acc']).mean()
ID_3_acc = (clustering_score['ID_3_filter'] == clustering_score['ID_3_acc']).mean()
n_clust_1 = len(clustering_score['ID_1_str'].unique())
n_clust_2 = len(clustering_score['ID_2_OPTICS'].unique())
n_clust_3 = len(clustering_score['ID_3_filter'].unique())
print(f'String feature clustering: {round(ID_1_acc, 5)}\n'f'Number of clusters: {n_clust_1}\n')
print(f'OPTICS clustering: {round(ID_2_acc, 5)}\n'f'Number of clusters: {n_clust_2}\n')
print(f'Filter clustering: {round(ID_3_acc, 5)}\n'f'Number of clusters: {n_clust_3}\n')


# In[20]: Exporting the labels to xlsx file
clustering_score.to_excel("clustering.xlsx")
