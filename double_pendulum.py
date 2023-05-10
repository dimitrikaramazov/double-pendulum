import math 
import numpy as np


class DoublePendulum:
    
    """This class defines a double pendulum system"""

    def __init__(self, teta1, teta2, teta1_dot = 0, teta2_dot = 0, g = 10, m = 0.01, l = 10, N = 100) :
        
        """
        Constructs a double pendulum with an initial condition.
        The first particle is the one connected to the fixed pivot

        teta1 - Initial angle of the first particle
        teta2 - Initial angle of the second particle
        teta1_dot - Initial angular velocity of the first particle
        teta2_dot - Initial angular velocity os the second particle
        g  - The gravitationa acceleration
        m  - The mass of each particle
        l  - the length of each rod
        N  - number of integration steps
        """
        self.g = g
        self.m = m
        self.l = l
        self.N = N
        
        #2 constats that apear in the equations of motion
        self.a = self.m * np.square(self.l)
        self.b = self.m * self.g * self.l
        
        # Compute the initial canonical momenta
        
        p1 = self.a * (2 * teta1_dot + teta2_dot * np.cos(teta1 - teta2))
        p2 = self.a *     (teta2_dot + teta1_dot * np.cos(teta1 - teta2))
        
        # the solution of the pendulum in phase space
        self.sol = np.zeros((self.N, 4))

        self.sol[0, :] = [teta1, teta2, p1, p2]

        # energy of the system at all times
        self.energy = np.zeros((self.N))    

        # position in the plane for the 2 particles
        self.r1 = np.zeros((self.N, 2))
        self.r2 = np.zeros((self.N, 2))
    def derivatives(self, X):
        """method that computes the derivative of a vector X from the phase space"""
        
        a = self.a
        b = self.b
        s = 1 + np.square(np.sin(X[0] - X[1]))
        cos = np.cos(X[0] - X[1])
        sin = np.sin(X[0] - X[1])
        sin2 = np.sin(2 * (X[0] - X[1]))

        teta1_dot = (X[2] - X[3] * cos) / (a * s)
        teta2_dot = (-X[2] * cos + 2 * X[3]) / (a * s)

        p1_dot =  sin2 * (np.square(X[2])/2 + np.square(X[3]) - X[2] * X[3] * cos) / (a * np.square(s)) - X[2] * X[3] * sin / (a * s) - 2 * b * np.sin(X[0])
        p2_dot = -sin2 * (np.square(X[2])/2 + np.square(X[3]) - X[2] * X[3] * cos) / (a * np.square(s)) + X[2] * X[3] * sin / (a * s) -     b * np.sin(X[1])

        return np.array([teta1_dot, teta2_dot, p1_dot, p2_dot])
    
    def hamiltonian(self,X):
        """ computes the hamiltonian function of the system for a given vector X """
        a = self.a
        b = self.b
        s = 1 + np.square(np.sin(X[0] - X[1]))
        cos = np.cos(X[0] - X[1])
        
        return  (np.square(X[2])/2 + np.square(X[3]) - X[2] * X[3] * cos) / (a * s) - b * (2 * np.cos(X[0]) + np.cos(X[1]))

    
    def compute_energy(self):
        """ method that computes the energy of the system at all times"""
        N = self.N
        for i in range(N): 
            self.energy[i] = self.hamiltonian(self.sol[i, :]) 
    
    def compute_r1(self):
        """computes the position in the plane of the first particle"""
        for i in range(self.N):
            self.r1[i,0] = self.l * np.sin(self.sol[i,0])
            self.r1[i,1] = -1 * self.l * np.cos(self.sol[i,0])
    
    def compute_r2(self):
        """computes the position in the plane of the second particle"""
        for i in range(self.N):
            self.r2[i,0] = self.l * (np.sin(self.sol[i,0]) + np.sin(self.sol[i,1]))
            self.r2[i,1] = -1 * self.l * (np.cos(self.sol[i,0]) + np.cos(self.sol[i,1]) )

    def compute_phase_space_difference(pen1, pen2):
        """probably we don't need this"""
        N = pen1.N
        delta_x = np.zeros((N, 4))
        
        d = pen1.sol[:,0] - pen2.sol[:,0]
        delta_x[:,0] = d - np.around(d/ (2 * np.pi), 0) * 2 * np.pi
        
        d = pen1.sol[:,1] - pen2.sol[:,1]
        delta_x[:,1] = d - np.around(d/ (2 * np.pi), 0) * 2 * np.pi
        
        delta_x[:,2] = pen1.sol[:,2] - pen2.sol[:,2]
        delta_x[:,3] = pen1.sol[:,3] - pen2.sol[:,3]
        return delta_x
    
    def jacobian(self,X):
        """ to be tested"""
        a = self.a
        b = self.b 

        J = np.zeros((4,4))
        # auxiliary functions
        s = 1 + np.square(np.sin(X[0] - X[1]))
        cos = np.cos(X[0] - X[1])
        sin = np.sin(X[0] - X[1])
        sin2 = np.sin(2 * (X[0] - X[1]))
        cos2 = np.cos(2 * (X[0] - X[1]))

        J[0,0] = sin * (-2 * X[2] + X[3] * ( 2 + np.square(cos))) / (a * np.square(s))
        J[0,1] = -1 * J[0,0]
        J[0,2] = 1/(a * s)
        J[0,3] = -1 * cos / (a * s)

        J[1,0] = sin * (X[2] * (2 + np.square(cos)) - 4 * X[3] * cos) / (a * np.square(s))
        J[1,1] = -1 * J[1,0]
        J[1,2] = J[0,3]
        J[1,3] = 2 * J[0,2]

        J[2,0] =  -2 * np.square(sin2) * (np.square(X[2])/2 + np.square(X[3]) - X[2] * X[3] * cos) / (a * np.power(s,3)) + (cos2 * (np.square(X[2]) + 2 * np.square(X[3])) - X[2] * X[3] * (2 * sin2 * sin - cos)) / (a * np.square(s)) -2 * b * np.cos(X[0])
        J[2,1] = -1 * (J[2,0] + 2 * b * np.cos(X[0]))
        J[2,2] = sin2 * (X[2] - X[3] * cos) / (a * np.square(s)) - X[3] * sin / ( a * s)
        J[2,3] = sin2 * (2 * X[3] - X[2] * cos) / (a * np.square(s)) - X[2] * sin / ( a * s)

        J[3,0] = -1 * J[2,0] - 2 * b * np.cos(X[1])
        J[3,1] = -1 * J[2,1] - b * np.cos(X[2])
        J[3,2] = -1 * J[2,2]
        J[3,3] = -1 * J[2,3]

        return J



    
    
