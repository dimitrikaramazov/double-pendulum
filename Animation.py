import numpy as np
import matplotlib.pyplot as plt
from numerical_methods import Numerical_Methods as nm
from double_pendulum import DoublePendulum
from matplotlib import animation as anim
from matplotlib.animation import PillowWriter

dt = 0.01

pen = DoublePendulum(2.001,1.01, N = 1500, l = 0.5)
pen2 = DoublePendulum(2,1, N = 1500, l = 0.5)


pen.sol = nm.rk4(pen.derivatives, pen.sol[0,:], dt, pen.N)
pen2.sol = nm.rk4(pen2.derivatives, pen2.sol[0,:], dt, pen2.N)

pen.compute_r1()
pen.compute_r2()
pen2.compute_r1()
pen2.compute_r2()
draw = plt.draw()


fig = plt.figure(figsize=(12,5))
x1 = pen.r1[:,0]
y1 = pen.r1[:,1]
x2 = pen.r2[:,0]
y2 = pen.r2[:,1]
x_1 = pen2.r1[:,0]
y_1 = pen2.r1[:,1]
x_2 = pen2.r2[:,0]
y_2 = pen2.r2[:,1]
fig,ax = plt.subplots()
plt.style.use("classic")

def animate(i):
    ax.clear()
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_title('movement of each pendulum')
    corde1, = ax.plot([0,x1[0]],[0,y1[0]], color = "black", lw = 1)
    corde2, = ax.plot([x1[0],x2[0]],[y1[0],y2[0]], color = "black", lw = 1)
    corde3, = ax.plot([0,x_1[0]],[0,y_1[0]], color = "grey", lw = 1)
    corde4, = ax.plot([x_1[0],x_2[0]],[y_1[0],y_2[0]], color = "grey", lw = 1)
    corde1.set_data([0,x1[i]],[0,y1[i]])
    corde2.set_data([x1[i],x2[i]],[y1[i],y2[i]])
    corde3.set_data([0,x_1[i]],[0,y_1[i]])
    corde4.set_data([x_1[i],x_2[i]],[y_1[i],y_2[i]])
    
    if i <= 10 : 
    
        line1, = ax.plot(x_2[0:i],y_2[0:i],color = "red", lw = 1)
        line2, = ax.plot(x2[0:i],y2[0:i],color = "blue", lw = 1)
        
    else : 
        line1, = ax.plot(x_2[i-10:i],y_2[i-10:i],color = "red", lw = 1)
        line2, = ax.plot(x2[i-10:i],y2[i-10:i],color = "blue", lw = 1)
    
    mass1, = ax.plot(x1[i], y1[i], marker='.', color='black')
    mass2, = ax.plot(x2[i], y2[i], marker='.', color='black')
    mass3, = ax.plot(x_1[i], y_1[i],marker='.', color='grey')
    mass4, = ax.plot(x_2[i], y_2[i],marker='.', color='grey')
    
    return line1, line2, mass1, mass2, mass3, mass4, corde1, corde2, corde3, corde4
ani = anim.FuncAnimation(fig, animate, interval = 10, blit = True,repeat = False, frames = 1000)
ani.save("Doublependulum.gif", dpi=300, writer=PillowWriter(fps=200))
