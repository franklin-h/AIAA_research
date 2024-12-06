import numpy as np
from scipy import integrate

array = [1,2,3]
np.array = np.zeros(4).tolist()

array = [array, np.array]

end_time = 10
dt = 0.001
N = np.floor(end_time/dt)
t_array = np.arange(0, N) * dt

f = lambda x, a: a*x
y = integrate.quad(f, 0, 10, args=(1,))


print(t_array)
print(y)
# print(array)
# print(np.array(array))