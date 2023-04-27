from double_pendulum import DoublePendulum

import matplotlib.pyplot as plt

from numerical_methods import Numerical_Methods as nm

dt = 0.001

pen = DoublePendulum(4.001,1.01, N = 20000, l = 0.5)

#print(pen.sol[0,:])

pen.sol = nm.euler_forward(pen.derivatives, pen.sol[0,:], dt, pen.N)

pen.compute_energy()

t = []
e = pen.hamiltonian(pen.sol[0,:])
E = []
for i in range(pen.N):
    t.append(dt * i)
    E.append(e)

pen.compute_r1()
pen.compute_r2()

plt.plot(pen.r1[:,0], pen.r1[:,1])    
plt.plot(pen.r2[:,0], pen.r2[:,1])
# plt.plot(t, pen.energy, color = 'red', label = 'Calculated Value')
# plt.plot(t,E, color = 'blue', label = 'Real Value')
# plt.xlabel("Time ( s )")
# plt.ylabel("Energy ( J )")
# plt.title("Graph of the energy for Euler Forward method")
# plt.legend(loc = 'best')
# plt.grid()
# plt.ylim(-1,1)

#plt.plot(pen.sol[:,0], pen.sol[:,2])