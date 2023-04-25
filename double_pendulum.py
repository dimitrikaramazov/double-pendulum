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
        """2 constats that apear in the equations of motion"""
        self.a = self.m * np.square(self.l)
        self.b = self.m * self.g * self.l
        
        """Compute the initial canonical momenta"""
        
        p1 = self.a * (2 * teta1_dot + teta2_dot * np.cos(teta1 - teta2))
        p2 = self.a *     (teta2_dot + teta1_dot * np.cos(teta1 - teta2))
        
        """The solution of the pendulum in phase space"""
        self.sol = np.zeros((self.N, 4))

        self.sol[0, :] = [teta1, teta2, teta1_dot, teta2_dot]

        self.energy = np.zeros((self.N))    

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

        p1 =  sin2 * (np.square(X[2])/2 + np.square(X[3]) - X[2] * X[3] * cos) / (a * np.square(s)) - X[2] * X[3] * sin / (a * s) - 2 * b * np.sin(X[0])
        p2 = -sin2 * (np.square(X[2])/2 + np.square(X[3]) - X[2] * X[3] * cos) / (a * np.square(s)) + X[2] * X[3] * sin / (a * s) -     b * np.sin(X[1])

        return np.array([teta1_dot, teta2_dot, p1, p2])
    
    def hamiltonian(self,X):
        
        a = self.a
        b = self.b
        s = 1 + np.square(np.sin(X[0] - X[1]))
        cos = np.cos(X[0] - X[1])
        
        return  (np.square(X[2])/2 + np.square(X[3]) - X[2] * X[3] * cos) / (a * s) - b * (2 * np.cos(X[0]) + np.cos(X[1]))

    
    def compute_energy(self):

        N = self.N
        for i in range(N): 
            self.energy[i] = self.hamiltonian(self.sol[i, :]) 
    
