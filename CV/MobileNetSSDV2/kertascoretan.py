import numpy as np
a=10-np.array([1,2,3,4,5])
print(a)
print(a**2)
print((a**2)**0.5)

b=np.array([[0,0],[1,2]])
b[0]=(1,2)
print(b)