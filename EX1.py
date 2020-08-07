
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic(u'matplotlib inline')


# In[23]:

lr_model = LinearRegression(normalize=True)


# In[155]:

#tedad 5 dade k=1
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
plt.xlabel("x")
plt.ylabel('y')
plt.title("k=1 and n = 5")
plt.show()


# In[156]:

#tedad 10 dade k=1
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
plt.xlabel("x")
plt.ylabel('y')
plt.title("k=1 and n = 10")
plt.show()


# In[157]:

#tedad 25 dade k=1
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
plt.xlabel("x")
plt.ylabel('y')
plt.title("k=1 and n = 25")
plt.show()


# In[158]:

#tedad 25 dade k=1
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
plt.xlabel("x")
plt.ylabel('y')
plt.title("k=1 and n = 25")
plt.show()

    


# In[159]:

#tedad 100 dade k=1
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
plt.xlabel("x")
plt.ylabel('y')
plt.title("k=1 and n = 100")
plt.show()


# In[160]:

#tedad dade 5 k = 4
x = np.random.rand(5, 1) * 20
x = np.hstack((x, x**2, x**3, x**4))
y = 2.358 * x[:,0] - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 4)
y = y.reshape(-1, 1)
lr_model.fit(x, y)
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4))
y_line = lr_model.predict(x_line)
plt.plot(x[:,0], y , 'bo')
plt.plot(x_line[:,0], y_line, 'r--')
plt.xlabel("x")
plt.ylabel('y')
plt.title("k=4 and n = 5")
plt.show()


# In[161]:

#tedad dade 10 k = 4
x = np.random.rand(10, 1) * 20
x = np.hstack((x, x**2, x**3, x**4))
y = 2.358 * x[:,0] - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 4)
y = y.reshape(-1, 1)
lr_model.fit(x, y)
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4))
y_line = lr_model.predict(x_line)
plt.plot(x[:,0], y , 'bo')
plt.plot(x_line[:,0], y_line, 'r--')
plt.xlabel("x")
plt.ylabel('y')
plt.title("k=4 and n = 10")
plt.show()


# In[162]:

#tedad dade 25 k = 4
x = np.random.rand(25, 1) * 20
x = np.hstack((x, x**2, x**3, x**4))
y = 2.358 * x[:,0] - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 4)
y = y.reshape(-1, 1)
lr_model.fit(x, y)
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4))
y_line = lr_model.predict(x_line)
plt.plot(x[:,0], y , 'bo')
plt.plot(x_line[:,0], y_line, 'r--')
plt.xlabel("x")
plt.ylabel('y')
plt.title("k=4 and n = 25")
plt.show()


# In[163]:

#tedad dade 100 k = 4
x = np.random.rand(100, 1) * 20
x = np.hstack((x, x**2, x**3, x**4))
y = 2.358 * x[:,0] - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 4)
y = y.reshape(-1, 1)
lr_model.fit(x, y)
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4))
y_line = lr_model.predict(x_line)
plt.plot(x[:,0], y , 'bo')
plt.plot(x_line[:,0], y_line, 'r--')
plt.xlabel("x")
plt.ylabel('y')
plt.title("k=4 and n = 100")
plt.show()


# In[164]:

#tedad dade 5 k = 16
x = np.random.rand(5, 1) * 20
x = np.hstack((x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13, x**14 , x**15 , x**16))
y = 2.358 * x[:,0] - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 16)
y = y.reshape(-1, 1)
lr_model.fit(x, y)
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4, x_line**5, x_line**6, x_line**7, x_line**8, x_line**9, x_line**10, x_line**11, x_line**12, x_line**13, x_line**14 , x_line**15 , x_line**16))
y_line = lr_model.predict(x_line)
plt.plot(x[:,0], y , 'bo')
plt.plot(x_line[:,0], y_line, 'r--')
plt.xlabel("x")
plt.ylabel('y')
plt.title("k=16 and n = 5")
plt.show()


# In[165]:

#tedad dade 10 k = 16
x = np.random.rand(10, 1) * 20
x = np.hstack((x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13, x**14 , x**15 , x**16))
y = 2.358 * x[:,0] - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 16)
y = y.reshape(-1, 1)
lr_model.fit(x, y)
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4, x_line**5, x_line**6, x_line**7, x_line**8, x_line**9, x_line**10, x_line**11, x_line**12, x_line**13, x_line**14 , x_line**15 , x_line**16))
y_line = lr_model.predict(x_line)
plt.plot(x[:,0], y , 'bo')
plt.plot(x_line[:,0], y_line, 'r--')
plt.xlabel("x")
plt.ylabel('y')
plt.title("k=16 and n = 10")
plt.show()


# In[166]:

#tedad dade 25 k = 16
x = np.random.rand(25, 1) * 20
x = np.hstack((x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13, x**14 , x**15 , x**16))
y = 2.358 * x[:,0] - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 16)
y = y.reshape(-1, 1)
lr_model.fit(x, y)
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4, x_line**5, x_line**6, x_line**7, x_line**8, x_line**9, x_line**10, x_line**11, x_line**12, x_line**13, x_line**14 , x_line**15 , x_line**16))
y_line = lr_model.predict(x_line)
plt.plot(x[:,0], y , 'bo')
plt.plot(x_line[:,0], y_line, 'r--')
plt.xlabel("x")
plt.ylabel('y')
plt.title("k=16 and n = 25")
plt.show()


# In[167]:

#tedad dade 100 k = 16
x = np.random.rand(100, 1) * 20
x = np.hstack((x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13, x**14 , x**15 , x**16))
y = 2.358 * x[:,0] - 3.121
y = y + np.random.normal(scale=3, size=y.shape)
x = x.reshape(-1, 16)
y = y.reshape(-1, 1)
lr_model.fit(x, y)
x_line = np.arange(0,20,0.1).reshape(-1,1)
x_line = np.hstack((x_line, x_line**2, x_line**3, x_line**4, x_line**5, x_line**6, x_line**7, x_line**8, x_line**9, x_line**10, x_line**11, x_line**12, x_line**13, x_line**14 , x_line**15 , x_line**16))
y_line = lr_model.predict(x_line)
plt.plot(x[:,0], y , 'bo')
plt.plot(x_line[:,0], y_line, 'r--')
plt.xlabel("x")
plt.ylabel('y')
plt.title("k=16 and n = 100")
plt.show()


# In[171]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression(fit_intercept=True, normalize=True)
#k=10
for count, i in enumerate([5, 10, 25, 100]):
    x = np.random.rand(i, 1) * 20
    x = x.reshape(-1, 1)
    y = 2.358 * x - 3.121
    y = y + np.random.normal(scale=3, size=y.shape)
    y = y.reshape(-1, 1)
    lr_model.fit(x, y)
    h = lr_model.predict(x)
    MSE = np.mean((y - h) ** 2) / 2
    print("MSE = ", 1 - (2 * MSE / y.var()))
    MSE_list = []
    MSE_list.append(MSE)
    for j in range(1, 10):
        x = np.hstack((x, x**(j+1)))
        y = 2.358 * x[0] - 3.121
        y = y + np.random.normal(scale=3, size=y.shape)
        y = y.reshape(-1, 1)
        lr_model.fit(x, y)
        h = lr_model.predict(x)
        MSE = np.mean((y-h)**2) / 2
        print("MSE = ", 1-(2*MSE / y.var()))
        MSE_list.append(MSE)
        plt.plot(range(1,11), MSE_list, 'bo')
        plt.xlabel("Power")
        plt.ylabel('MSE')
        plt.show()
    


# In[170]:




# In[121]:



