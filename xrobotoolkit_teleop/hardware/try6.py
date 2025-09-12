import numpy as np
a=np.array([2,3,4,5,6,9])

print(a)
b=np.array([1,2,1,3,1,6])
a=np.clip(a,-b,b)
print(a)