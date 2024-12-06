from scipy.integrate import odeint
from scipy import integrate
import numpy as np

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

    t_array = np.arange(0, N) * dt

    # Velocity from release to fully unfurled
    if end_time < t_unfurl:
        v_z_output = g * end_time
    else:
        v_0 = g * t_unfurl
        v_z_output = (-alpha*v_0 - np.tanh(end_time*alpha*g))/(np.tanh(end_time*alpha*g)*alpha**2 * v_0 - alpha)
    return v_z_output


    # index_t_unfurl = closest_index(t_array,t_unfurl)
    # v_z_init_phase = g * t_array[0:index_t_unfurl]
    # v_0 = v_z_init_phase[-1]
    #
    # v_z_unfurled = np.zeros(len(t_array) - len(v_z_init_phase))
    #
    # for t in t_array[index_t_unfurl+1:-1]-t_array[index_t_unfurl+1]:
    #     v_z_unfurled[t] = (-alpha*v_0 - np.tanh(t*alpha*g))/(np.tanh(t*alpha*g)*alpha^2 * v_0 - alpha)


    # v_z_concatenated = [*v_z_init_phase, *v_z_unfurled]
    # return v_z_concatenated

def waypoint_correction(z,u_z1,u_z2,z1,z2,t_unfurl,end_time,d,K,rho,A,m,d,dt):

    '''
    K is the von Karman constant.
    z is the current height of the drone
    z1 and z2 are two altitudes where wind was measured
    u_z1 and u_z2 are corresponding wind measurement.
    d is characteristic height depending on terrain.
    rho is air density at sea level (good enough approximation)
    A is cross-sectional area of payload/parachute system
    m is mass of payload/parachute system.
    d is zero plane displacement (height of the averager obstacle impeding wind flow
    dt specify the timestep for numerical operations.
    '''

    # Calculate the frictional velocity
    u_star = K*(u_z2 - u_z1)/np.log((z2-d)/(z1-d))
    v_z = v_z(A,C_d,m,g,rho,t_unfurl,dt,end_time)

    # Integrate vertical velocity to get vertical displacement
    # payload_height = integrate.qu
    z = integrate.simpson()

    u_z = (u_star/K) * ln()
    def dvdt(t,v,u_z):
        dvdt = (0.5*rho*A/m) * (z - d)

