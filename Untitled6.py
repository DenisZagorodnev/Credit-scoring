
# coding: utf-8

# In[118]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#get_ipython().magic('matplotlib inline')


# In[119]:


data = pd.read_csv('data.txt', header=None, na_values='?') 
#загрузил данные, без заголовков, пропщенные значения заменены ?


# In[120]:


data.shape
#размер таблицы


# In[121]:


data.columns = ['A' + str(i) for i in range(1, 16)] + ['class']
#задал имена признаков и результат


# In[122]:


categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']
print (categorical_columns)
print (numerical_columns)
#выделение числовых и категориальных признаков


# In[123]:


from pandas.plotting import scatter_matrix
scatter_matrix(data, alpha=0.05, figsize=(10, 10));
#корреляционная матрица признаков


# In[124]:


data.corr()
#корреляция признаков практически отстуствует


# In[125]:


data = data.fillna(data.median(axis=0), axis=0)
#заполнили пропуски медианными значениями


# In[126]:


data['A1'].describe()


# In[127]:


data['A1'] = data['A1'].fillna('b')
data['A1'].describe()


# In[128]:


data_describe = data.describe(include=[object])
for c in categorical_columns:
    data[c] = data[c].fillna(data_describe[c]['top'])
    #категориальные пропуск заполнили топовым значением


# In[129]:


binary_columns    = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
print (binary_columns, nonbinary_columns)
#выделили бинарные и небинарные признаки


# In[130]:


data_describe = data.describe(include=[object])
data_describe


# In[131]:


for c in binary_columns[0:]:
    top = data_describe[c]['top']
    top_items = data[c] == top
    data.loc[top_items, c] = 0
    data.loc[np.logical_not(top_items), c] = 1
    #заменили бинарные на 0/1


# In[132]:


data[binary_columns].describe()


# In[133]:


data_nonbinary = pd.get_dummies(data[nonbinary_columns])
print(data_nonbinary.columns)
#небинарные делим на несколько признаков и сводим к бинарным каждый отдельно


# In[134]:


data_numerical = data[numerical_columns]
data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()
data_numerical.describe()
#нормализация количественных данных


# In[135]:


data = pd.concat((data_numerical, data[binary_columns], data_nonbinary), axis=1)
data = pd.DataFrame(data, dtype=float)
print (data.shape)
print(data.columns)
#соединяем все данные вместе в одну таблицу


# In[136]:


X = data.drop(('class'), axis=1)  # отделяем вектор-столбец 'class'.
y = data['class']
feature_names = X.columns
print (feature_names)


# In[137]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 11)

N_train, _ = X_train.shape 
N_test,  _ = X_test.shape 
print (N_train, N_test)
#разбили таблицу на тестовую и обучающие выборки

# In[138]:


from sklearn import ensemble
rf = ensemble.RandomForestClassifier(n_estimators=1500, random_state=11)
rf.fit(X_train, y_train)

err_train = np.mean(y_train != rf.predict(X_train))
err_test  = np.mean(y_test  != rf.predict(X_test))
print (err_train, err_test)
#алгоритм "случайный лес"


# In[139]:


importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature importances:")
for f, idx in enumerate(indices):
    print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))
    #значимость отдельных признаков


# In[140]:


d_first = 20
plt.figure(figsize=(8, 8))
plt.title("Feature importances")
plt.bar(range(d_first), importances[indices[:d_first]], align='center')
plt.xticks(range(d_first), np.array(feature_names)[indices[:d_first]], rotation=90)
plt.xlim([-1, d_first]);
#таблица значимости


# In[141]:


best_features = indices[:16]
best_features_names = feature_names[best_features]
print(best_features_names)
# 16 наиболее значимых


# In[142]:


from sklearn import ensemble
rf = ensemble.RandomForestClassifier(n_estimators=1500, random_state=11)
rf.fit(X_train[best_features_names], y_train)

err_train = np.mean(y_train != rf.predict(X_train[best_features_names]))
err_test  = np.mean(y_test  != rf.predict(X_test[best_features_names]))
print (err_train, err_test)

