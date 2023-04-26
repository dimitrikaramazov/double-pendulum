import numpy as np

from double_pendulum import DoublePendulum

import matplotlib.pyplot as plt

from numerical_methods import Numerical_Methods as nm

dt = 0.01
# def Eu_ap(F,y0,Dt,N) : #Forward Euler, F is the function, y0 the initial condition, Dt the time interval and N is the number of iteration.
#     y = []
#     for i in range(N):
        
#         if i == 0:
#             y.append(y0)
#         else :
#             y_i = y[i-1] + F(y[i-1])*Dt
#             y.append(y_i)
#     return np.array(y)

# def RK4(F,y0,Dt,N): #RK4 method, same arguments
#     y = []
#     for i in range(N):
        
#         if i == 0:
#             y.append(y0)
#         else :
#             k_1 = F(y[i-1])*Dt
#             k_2 = F(y[i-1]+k_1/2)*Dt
#             k_3 = F(y[i-1]+k_2/2)*Dt
#             k_4 = F(y[i-1]+k_3)*Dt
#             y_i = y[i-1] + (k_1 + 2*k_2 + 2*k_3 + k_4) / 6
#             y.append(y_i)
#     return np.array(y)

pen = DoublePendulum(3,2, N = 10000, l = 0.5)

#print(pen.sol[0,:])

pen.sol = nm.euler_forward(pen.derivatives, pen.sol[0,:], dt, pen.N)

pen.compute_energy()

t = []
e = pen.hamiltonian(pen.sol[0,:])
E = []
for i in range(pen.N):
    t.append(dt * i)
    E.append(e)
    
plt.plot(t, pen.energy, color = 'red', label = 'Calculated Value')
plt.plot(t,E, color = 'blue', label = 'Real Value')
plt.xlabel("Time ( s )")
plt.ylabel("Energy ( J )")
plt.title("Graph of the energy for Euler Forward method")
plt.legend(loc = 'best')
plt.grid()
plt.ylim(-1,1)

#plt.plot(pen.sol[:,0], pen.sol[:,2])
