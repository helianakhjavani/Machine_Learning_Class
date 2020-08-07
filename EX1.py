
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic(u'matplotlib inline')


# In[23]:

lr_model = LinearRegression(normalize=True)


# In[45]:

#tedad 5 dade
x = np.random.rand(5, 1) * 20
y = 2.358 * x - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
lr_model.fit(x, y)
x_line = np.random.rand(100, 1) * 20
y_line = lr_model.coef_ * x_line + lr_model.intercept_
plt.plot(x_line, y_line, 'r--')
plt.plot(x, y, 'bo')


# In[46]:

#tedad 10 dade
x = np.random.rand(10, 1) * 20
y = 2.358 * x - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
lr_model.fit(x, y)
x_line = np.random.rand(100, 1) * 20
y_line = lr_model.coef_ * x_line + lr_model.intercept_
plt.plot(x_line, y_line, 'r--')
plt.plot(x, y, 'bo')


# In[104]:

#tedad 5 dade
x = np.random.rand(25, 1) * 20
y = 2.358 * x - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
lr_model.fit(x, y)
x_line = np.random.rand(100, 1) * 20
y_line = lr_model.coef_ * x_line + lr_model.intercept_
plt.plot(x_line, y_line, 'r--')
plt.plot(x, y, 'bo')


# In[47]:

#tedad 5 dade
x = np.random.rand(25, 1) * 20
y = 2.358 * x - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
lr_model.fit(x, y)
x_line = np.random.rand(100, 1) * 20
y_line = lr_model.coef_ * x_line + lr_model.intercept_
plt.plot(x_line, y_line, 'r--')
plt.plot(x, y, 'bo')
    


# In[58]:

#tedad 5 dade
x = np.random.rand(100, 1) * 20
y = 2.358 * x - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
lr_model.fit(x, y)
x_line = np.random.rand(100, 1) * 20
y_line = lr_model.coef_ * x_line + lr_model.intercept_
plt.plot(x_line, y_line, 'r--')
plt.plot(x, y, 'bo')
    


# In[105]:

#tedad dade 5 k = 4
x = np.random.rand(5, 1) * 20
x = np.hstack((x, x**2, x**3, x**4))
y = 2.358 * x - 3.121
y = y + np.random.normal(scale=5, size=y.shape)
x = x.reshape(-1, 4)
y = y.reshape(-1, 4)
lr_model.fit(x, y)
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4))
y_line = lr_model.predict(x_line)
plt.plot(x[:,0], y , 'bo')
plt.plot(x_line[:,0], y_line, 'r--')


# In[106]:

#tedad dade 19 k = 4
x = np.random.rand(10, 1) * 20
x = np.hstack((x, x**2, x**3, x**4))
y = 2.358 * x - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 4)
y = y.reshape(-1, 4)
lr_model.fit(x, y)
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4))
y_line = lr_model.predict(x_line)
plt.plot(x[:,0], y , 'bo')
plt.plot(x_line[:,0], y_line, 'r--')


# In[67]:

#tedad dade 25 k = 4
x = np.random.rand(25, 1) * 20
x = np.hstack((x, x**2, x**3, x**4))
y = 2.358 * x - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 4)
y = y.reshape(-1, 4)
lr_model.fit(x, y)
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4))
y_line = lr_model.predict(x_line)
plt.plot(x[:,0], y , 'bo')
plt.plot(x_line[:,0], y_line, 'r--')


# In[69]:

#tedad dade 100 k = 4
x = np.random.rand(100, 1) * 20
x = np.hstack((x, x**2, x**3, x**4))
y = 2.358 * x - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 4)
y = y.reshape(-1, 4)
lr_model.fit(x, y)
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4))
y_line = lr_model.predict(x_line)
plt.plot(x[:,0], y , 'bo')
plt.plot(x_line[:,0], y_line, 'r--')


# In[71]:

#tedad dade 5 k = 16
x = np.random.rand(5, 1) * 20
x = np.hstack((x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13, x**14 , x**15 , x**16))
y = 2.358 * x - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 16)
y = y.reshape(-1, 16)
lr_model.fit(x, y)
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4, x_line**5, x_line**6, x_line**7, x_line**8, x_line**9, x_line**10, x_line**11, x_line**12, x_line**13, x_line**14 , x_line**15 , x_line**16))
y_line = lr_model.predict(x_line)
plt.plot(x[:,0], y , 'bo')
plt.plot(x_line[:,0], y_line, 'r--')


# In[72]:

#tedad dade 5 k = 16
x = np.random.rand(10, 1) * 20
x = np.hstack((x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13, x**14 , x**15 , x**16))
y = 2.358 * x - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 16)
y = y.reshape(-1, 16)
lr_model.fit(x, y)
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4, x_line**5, x_line**6, x_line**7, x_line**8, x_line**9, x_line**10, x_line**11, x_line**12, x_line**13, x_line**14 , x_line**15 , x_line**16))
y_line = lr_model.predict(x_line)
plt.plot(x[:,0], y , 'bo')
plt.plot(x_line[:,0], y_line, 'r--')


# In[73]:

#tedad dade 5 k = 16
x = np.random.rand(25, 1) * 20
x = np.hstack((x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13, x**14 , x**15 , x**16))
y = 2.358 * x - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 16)
y = y.reshape(-1, 16)
lr_model.fit(x, y)
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4, x_line**5, x_line**6, x_line**7, x_line**8, x_line**9, x_line**10, x_line**11, x_line**12, x_line**13, x_line**14 , x_line**15 , x_line**16))
y_line = lr_model.predict(x_line)
plt.plot(x[:,0], y , 'bo')
plt.plot(x_line[:,0], y_line, 'r--')


# In[74]:

#tedad dade 5 k = 16
x = np.random.rand(100, 1) * 20
x = np.hstack((x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13, x**14 , x**15 , x**16))
y = 2.358 * x - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 16)
y = y.reshape(-1, 16)
lr_model.fit(x, y)
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4, x_line**5, x_line**6, x_line**7, x_line**8, x_line**9, x_line**10, x_line**11, x_line**12, x_line**13, x_line**14 , x_line**15 , x_line**16))
y_line = lr_model.predict(x_line)
plt.plot(x[:,0], y , 'bo')
plt.plot(x_line[:,0], y_line, 'r--')


# In[112]:

def model(m,k):
    x = np.random.rand(m, 1) * 20
    for i in range(k-1):
        x = np.hstack((x, x**(i+2)))
    y = 2.358 * x - 3.121
    y = y + np.random.normal(scale=3, size=y.shape)
    lr_model.fit(x, y)
    #x_line = np.arange(0,20,0.1).reshape(-1,1)
    #for i in range(k-1):
     #   x_line = np.hstack((x_line, x_line**(i+2)))
    #y_line = lr_model.predict(x_line)
   # mse = np.mean((y_line - y) ** 2) / 2
    plt.plot(x[:,0], y , 'bo')
    #plt.plot(x_line[:,0], y_line, 'r--')
   #return mse

    


# In[115]:

model(100, 10)

