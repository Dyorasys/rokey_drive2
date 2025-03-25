import numpy as np
c = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,1,1,1]])
a = np.array([[1, 1, 2,2]])
print(c.dot(a.T)[0])