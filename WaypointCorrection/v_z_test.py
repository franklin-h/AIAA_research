from scipy.integrate import odeint
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

def closest_index(arr, target):
    # Use NumPy to calculate the absolute differences
    arr = np.array(arr)
    index = np.argmin(np.abs(arr - target))  # Find index of minimum difference
    return index

# Define vertical velocity of payload as function of time.
def v_z(A,C_d,m,g,rho,t_unfurl,end_time,dt=0.01):

    '''
    t_unfurl: The time it takes for the parachute to fully unfurl. Neglect Drag due to small cross sectional area.
    '''
    alpha = rho*A*C_d/(2*m*g)

    # dt = 1.25
    N = np.floor(end_time/dt)

    # Velocity from release to fully unfurled
    if end_time < t_unfurl:
        v_z_output = -g * end_time
    else:
        v_0 = -g * t_unfurl
        # print(v_0)
        end_time -= t_unfurl
        v_z_output = (-alpha*v_0 - np.tanh(end_time*alpha*g))/(np.tanh(end_time*alpha*g)*alpha**2 * v_0 - alpha)
    return v_z_output
t = np.arange(0,10,0.01).tolist()

A = 1.0  # Cross-sectional area of parachute
C_d = 1.4701034  # Drag coefficient
m = 0.236  # Mass in kg
g = 9.81  # Gravitational acceleration in m/s^2
rho = 1.225  # Air density in kg/m^3
t_unfurl = 0.3  # Time for parachute to fully unfurl in seconds

# Apply v_z function to all values of t
# t_list = t.tolist()
v_z_values = np.array([v_z(A, C_d, m, g, rho, t_unfurl, t_i) for t_i in t])

plt.plot(t,v_z_values)
# plt.xlim(0.85,1.1)
plt.show()
