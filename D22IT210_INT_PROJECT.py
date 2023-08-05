#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler=MinMaxScaler()


# In[2]:


df=pd.read_csv(r"C:\Users\yashb.DESKTOP-OQFJR2D\Downloads\archive.zip")



# In[3]:


df.shape


# In[4]:


df.head(10)


# In[5]:


df.info()


# In[6]:


df.isna().sum()


# In[7]:


from sklearn.preprocessing import LabelEncoder

# Create a new DataFrame with the desired columns
data_df = pd.DataFrame(df, columns =['pid','age', 'meno', 'size', 'grade', 'nodes', 'pgr', 'er', 'hormon', 'rfstime', 'status'])

# Apply label encoding to each column
for column in data_df.columns:
    label_encoder = LabelEncoder()
    data_df[column] = label_encoder.fit_transform(data_df[column])

# Print the updated DataFrame
print(data_df)



# In[8]:


data_df.head(5)


# In[9]:


categorical_cols = ['pid', 'age', 'meno', 'size', 'grade', 'nodes', 'pgr', 'er', 'hormon', 'rfstime']

# Perform one-hot encoding
encoded_df = pd.get_dummies(data_df, columns=categorical_cols,dtype=int)

# Print the encoded DataFrame
print(encoded_df)



# In[45]:


corr_matrix = df.corr().abs()

filtered_corr_df = corr_matrix

plt.figure(figsize=(10,10))
sns.heatmap(filtered_corr_df, annot=True, cmap="Reds")
plt.show()


# In[22]:


# In[27]:


X_train.shape,y_train.shape



# In[44]:


# In[28]:


from sklearn.preprocessing import StandardScaler



# In[24]:


# In[29]:


df.describe()



# In[25]:


# In[30]:


scaler=StandardScaler()



# In[26]:


# In[31]:


scaler.fit(X_train)


# In[32]:


x_train_scaler=scaler.transform(X_train)
x_test_scaler=scaler.transform(X_test)


# In[33]:


x_train_scaler


# In[34]:


x_test_scaler



# In[28]:


# In[36]:


x_train_scaler=pd.DataFrame(x_train_scaler,columns=X_train.columns)


# In[37]:


x_train_scaler


# # In[38]:


# x_test_scaler=pd.DataFrame(x_test_scaler,columns=X_test.columns)



# In[29]:


# In[38]:


x_test_scaler=pd.DataFrame(x_test_scaler,columns=X_test.columns)


# In[30]:


# In[38]:


x_test_scaler=pd.DataFrame(x_test_scaler,columns=X_test.columns)


# In[39]:


x_test_scaler


# In[31]:


# In[40]:


x_test_scaler.describe()



# In[32]:


# In[41]:


x_train_scaler.describe()



# In[33]:


# In[42]:


np.round(x_train_scaler.describe(),1)



# In[34]:


# In[46]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



# In[35]:


# In[47]:


y_pred



# In[36]:


# In[48]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error



# In[37]:


# In[49]:


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)



# In[38]:


# In[51]:


rmse=np.sqrt(mean_squared_error(y_test,y_pred))



# In[39]:


# In[52]:


rmse




# In[40]:


# In[53]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots()

# Plot the actual values
ax.scatter(range(len(y_test)), y_test, color='blue', label='Actual')


# In[41]:


# In[53]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots()

# Plot the actual values
ax.scatter(range(len(y_test)), y_test, color='blue', label='Actual')

# Plot the predicted values
ax.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted')

# Set labels and title
ax.set_xlabel('Data Point')
ax.set_ylabel('Value')
ax.set_title('Actual vs. Predicted Values')

# Add a legend
ax.legend()

# Display the plot
plt.show()



# In[ ]:




