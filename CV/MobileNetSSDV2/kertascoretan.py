import numpy as np
a=10-np.array([1,2,3,4,5])
print(a)
print(a**2)
print((a**2)**0.5)

b=np.array([[0,0],[1,2],[2,2]])
d=np.array([[2,2],[1,2],[2,2]])
c=np.array([-5,-5])
print(len(b))
print(([1,2]-b)**2)
print((([1,2]-b)**2).sum(axis=1))
print(a[-1])