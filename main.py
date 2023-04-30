import numpy as np
import matplotlib.pyplot as plt
from numerical_methods import Numerical_Methods as nm
from double_pendulum import DoublePendulum
from matplotlib import animation as anim
from matplotlib.animation import PillowWriter
import networkx as nx

dt = 0.01

pen = DoublePendulum(2.001,1.01, N = 20000, l = 0.5)

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
draw = plt.draw()
# plt.plot(pen.r1[:,0], pen.r1[:,1])    
# plt.plot(pen.r2[:,0], pen.r2[:,1])


fig = plt.figure(figsize=(12,5))
x1 = pen.r1[:,0]
y1 = pen.r1[:,1]
x2 = pen.r2[:,0]
y2 = pen.r2[:,1]
fig,ax = plt.subplots()
plt.style.use("seaborn-v0_8-whitegrid")

def animate(i):
    ax.clear()
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_title('movement of each pendulum')
    
    line1, = ax.plot(x1[0:i],y1[0:i],color = "red", lw = 1)
    line2, = ax.plot(x2[0:i],y2[0:i],color = "blue", lw = 1)
    
    mass1, = ax.plot(x1[i], y1[i], marker='.', color='black')
    mass2, = ax.plot(x2[i], y2[i], marker='.', color='black')
    
    return line1, line2, mass1, mass2,
ani = anim.FuncAnimation(fig, animate, interval = 400, blit = True,repeat = False, frames = 1000)
ani.save("Doublependulum.gif", dpi=300, writer=PillowWriter(fps=200))


# plt.plot(t, pen.energy, color = 'red', label = 'Calculated Value')
# plt.plot(t,E, color = 'blue', label = 'Real Value')
# plt.xlabel("Time ( s )")
# plt.ylabel("Energy ( J )")
# plt.title("Graph of the energy for Euler Forward method")
# plt.legend(loc = 'best')
# plt.grid()
# plt.ylim(-1,1)

#plt.plot(pen.sol[:,0], pen.sol[:,2])
