import numpy as np
import matplotlib.pyplot as plt
from numerical_methods import Numerical_Methods as nm
from double_pendulum import DoublePendulum
from matplotlib import animation as anim
from matplotlib.animation import PillowWriter

dt = 0.01
theta = np.pi/40
pen = DoublePendulum(2,1, N = 6000, l = 0.5)
pen2 = DoublePendulum(2.01, 2.01, N = 3000, l = 0.5)

pen.sol = nm.rk4(pen.derivatives, pen.sol[0,:], dt, pen.N)
pen2.sol = nm.rk4(pen2.derivatives, pen2.sol[0,:], dt, pen2.N)


pen.compute_r1()
pen.compute_r2()
pen2.compute_r1()
pen2.compute_r2()
pen.compute_energy()

t = []
e = pen.hamiltonian(pen.sol[0,:])
E = []
for i in range(pen.N):
    t.append(dt * i)
    E.append(e)
    
#For each pendulum, we plot theta and the impulsion :
if True :
    plt.plot(pen.sol[:,1],pen.sol[:,3], color = 'red')
    plt.title('Phase space for the first mass')
    plt.xlabel('Theta')
    plt.ylabel('impulsion')

if False :
    plt.plot(pen.sol[:,0],pen.sol[:,2], color = 'blue')
    plt.xlabel('Theta')
    plt.title('Phase space for the second mass')
    plt.ylabel('impulsion')

#Plot for the positions of the two pendulum

if False :
    
    plt.plot(pen.r2[:,0], pen.r2[:,1], color = 'blue')
    plt.plot(pen2.r2[:,0], pen2.r2[:,1], color = 'red')
    plt.title('Position of the two pendulum')
    plt.grid()

    #plt.savefig("2_pen_traj.png")


#Plot for the energy of the system

if False :
    
    plt.plot(t, pen.energy, color = 'red', label = 'Calculated Value')
    plt.plot(t,E, color = 'blue', label = 'Real Value')
    plt.xlabel("Time ( s )")
    plt.ylabel("Energy ( J )")
    plt.title("Graph of the energy for Euler Forward method")
    plt.legend(loc = 'best')
    plt.grid()
    plt.ylim(-1,1)
 
# lyapunov coefficients computations 

if False :
    pen = DoublePendulum(0,theta, N = 6000, l = 0.5)
    pen.sol = nm.rk4(pen.derivatives, pen.sol[0,:], dt, pen.N)
    delta_x = np.zeros_like(pen.sol)
    delta_x[0,:] = [0.01,0.02,0,0]

    delta_x = nm.rk4_lyapunov_normalised(pen.jacobian, pen.sol, delta_x[0,:], dt, pen.N, 50)




    temp = nm.normalize_vector(delta_x[pen.N - 1,:])


    # initalising a new pendulum 

    pen = DoublePendulum(0, theta, N = 6000, l = 0.5)
    pen.sol = nm.rk4(pen.derivatives, pen.sol[0,:], dt, pen.N)

    delta_x = np.zeros_like(pen.sol)
    delta_x[0,:] = temp

    delta_x = nm.rk4_lyapunov_normalised(pen.jacobian, pen.sol, delta_x[0,:], dt, pen.N,50)

    lambda_max = np.zeros(pen.N)

    delta_x0_norm = np.linalg.norm(delta_x[0,:])
    for j in range(1, pen.N):
        lambda_max[j] = np.log(np.linalg.norm(delta_x[j,:])) / j*dt
        
    #Lambda Max plot relative to time
    
    plt.plot(t, lambda_max)
    plt.title('Lambda max')
    plt.xlabel('Time(t)')
    plt.ylabel('Lambda max')

    plt.grid()
    plt.savefig("lambda_max.png")
    print(lambda_max[pen.N-1])

# Create a figure with four subplots for the delta x

if False :
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))

    # Plot each curve on a separate subplot
    axs[0,0].plot(t, delta_x[:,0],color = 'red')
    axs[0,0].set_title('tetha1')
    axs[0,1].plot(t, delta_x[:,1],color = 'blue')
    axs[0,1].set_title('tetha2')
    axs[1,0].plot(t, delta_x[:,2], color = 'green')
    axs[1,0].set_title('p1')
    axs[1,1].plot(t, delta_x[:,3],color = 'black')
    axs[1,1].set_title('p2')

    # Add a shared x-axis label and y-axis label
    fig.text(0.5, 0.04, 'Time', ha='center', va='center')
    fig.text(0.06, 0.5, 'Delta X', ha='center', va='center', rotation='vertical')

    # Add some padding between subplots
    fig.subplots_adjust(hspace=0.4)

    # Save the figure to a PNG file
    fig.savefig('delta_x_1000.png', dpi=300)


# plt.plot(t, pen.energy, color = 'red', label = 'Calculated Value')
# plt.plot(t,E, color = 'blue', label = 'Real Value')
# plt.xlabel("Time ( s )")
# plt.ylabel("Energy ( J )")
# plt.title("Graph of the energy for Euler Forward method")
# plt.legend(loc = 'best')
# plt.grid()
# plt.ylim(-1,1)

#plt.plot(pen.sol[:,0], pen.sol[:,2])
