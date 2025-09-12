import numpy as np
a=np.arange(12)
a=np.clip(a,2,8)
print(a)
b=np.ones(12)
c=(a-b)/0.01
print(c)