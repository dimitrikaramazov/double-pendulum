import numpy as np
class Numerical_Methods :

    import numpy as np

    def normalize_vector(v):
        """
        Given a  NumPy array v, this function returns the
        same vector normalized
        """
        norm = np.linalg.norm(v)
        if norm == 0:
                return v
        return v / norm
    
    def euler_forward(F,y0,Dt,N) : #Forward Euler, F is the function, y0 the initial condition, Dt the time interval and N is the number of iteration.
        y = []
        for i in range(N):
        
            if i == 0:
                y.append(y0)
            else :
                y_i = y[i-1] + F(y[i-1])*Dt
                y.append(y_i)
        return np.array(y) 

    def euler_backward(F,y0,Dt,N) : #Backward Euler, same arguments as Eu_ap
        y = []
        for i in range(N):
        
            if i == 0:
                y.append(y0)
            else :
                y_inter = y[i-1] + F(y[i-1])*Dt
                y_i = y[i-1] + F(y_inter)*Dt
                y.append(y_i)
            
        return np.array(y)
            
    def euler_centered(F,y0,Dt,N): #Centered Euler, same arguments as Eu_ap
        y = []
        for i in range(N):
        
            if i == 0:
                y.append(y0)
            else :
                y_inter = y[i-1] + F(y[i-1])*Dt
                y_i = y[i-1] + 1/2*(F(y[i-1])+F(y_inter))*Dt
                y.append(y_i)
        return np.array(y)

    def rk4(F,y0,Dt,N): #RK4 method, same arguments
        y = []
        for i in range(N):
        
            if i == 0:
                y.append(y0)
            else :
                k_1 = F(y[i-1])*Dt
                k_2 = F(y[i-1]+k_1/2)*Dt
                k_3 = F(y[i-1]+k_2/2)*Dt
                k_4 = F(y[i-1]+k_3)*Dt
                y_i = y[i-1] + (k_1 + 2*k_2 + 2*k_3 + k_4) / 6
                y.append(y_i)
        return np.array(y)
    
    def rk4_lyapunov(J,y,dy0,Dt,N):
        """to be tested not sure yet"""
        dy = []
        for i in range(N):
        
            if i == 0:
                dy.append(dy0)
            else :
                k_1 = J(y[i-1]).dot(dy[i-1])*Dt
                k_2 = J(y[i-1]).dot(dy[i-1] + k_1/2)*Dt
                k_3 = J(y[i-1]).dot(dy[i-1] + k_2/2)*Dt
                k_4 = J(y[i]).dot(dy[i-1] + k_3)*Dt
                dy_i = dy[i-1] + (k_1 + 2*k_2 + 2*k_3 + k_4) / 6
                dy.append(dy_i)
        return np.array(dy)

    def rk4_lyapunov_normalised(J,y,dy0,Dt,N, normalisation_step):
        """to be tested not sure yet"""
        dy = []
        n = normalisation_step
        for i in range(N):
        
            if i == 0:
                dy.append(dy0)
            else :
                k_1 = J(y[i-1]).dot(dy[i-1])*Dt
                k_2 = J(y[i-1]).dot(dy[i-1] + k_1/2)*Dt
                k_3 = J(y[i-1]).dot(dy[i-1] + k_2/2)*Dt
                k_4 = J(y[i]).dot(dy[i-1] + k_3)*Dt
                dy_i = dy[i-1] + (k_1 + 2*k_2 + 2*k_3 + k_4) / 6
                if i % n == 0:
                    dy.append(Numerical_Methods.normalize_vector(dy_i))
                else:
                    dy.append(dy_i)
        return np.array(dy)
