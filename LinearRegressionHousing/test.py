import numpy as np
import matplotlib.pyplot as plt
import tkinter
import matplotlib
matplotlib.use('TkAgg')


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

#print(X)

X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
#print(X_b)

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

#print(theta_best)

X_new = np.array([[0], [2]])
print(X_new)

X_test = np.c_[np.ones((2, 2)), X_new] # add x0 = 1 to each instance
print(X_test)


X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
print(X_new_b)

y_predict = X_new_b.dot(theta_best)
print(y_predict)

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
#plt.savefig("mygraph.png")
plt.show()

