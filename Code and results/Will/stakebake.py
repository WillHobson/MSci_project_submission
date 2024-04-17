"""In this python script are the functions needed to run the time
dependent Stakebake model"""


"""
dependencies
"""
import numpy as np
import math


class SB():
        """
    stakebake is an implementation of the equation given J. L. Stakebake's paper ‘Kinetics for the reaction of hydrogen with uranium powder’
    INPUT: reaction rate constant, K (/s)
           reaction order, n (0<n<1)
    OUTPUT: the amount of hydrogen reacted at time t, wt (mg/g)
    """
    def stakebake(K, wm, n):
        if n != 1:
            Mo = 1 # in mg/g
            M = Mo**(n-1)

            #calclaute upper limit of t based on stakebake paper in min
            upper_t_lim = (Mo**(1-n))/(K*(1-n))
            upper_t_lim = round_decimals_down(upper_t_lim, 2) # round to 2 dp

            #create array of times between 0 and upper lim in steps of 0.1
            t=np.linspace(0,int(upper_t_lim),int(upper_t_lim*100000+1))
            wt = (1- (1+(K*(n-1)*M*t))**(1/(1-n)) )*wm # calculate wm
        else:
            t = np.linspace(0,15, 150)
            wt = wm*(1- np.exp(-K*t))
        return wt, t
    
    def round_decimals_down(number:float, decimals:int=2):
        """
        Returns a value rounded down to a specific number of decimal places.
        """
        if not isinstance(decimals, int):
            raise TypeError("decimal places must be an integer")
        elif decimals < 0:
            raise ValueError("decimal places has to be 0 or more")
        elif decimals == 0:
            return math.floor(number)

        factor = 10 ** decimals
        return math.floor(number * factor) / factor
    
    def optimize_stakebake(xdata, ydata, k=None, n=None):
        from scipy.optimize import minimize
        from sklearn.metrics import r2_score
        def your_model_function(params, x_data):
            K, n = params
            wm=12.5
            t = x_data
            Mo =1
            
            if n != 1:
                
                #calclaute upper limit of t based on stakebake paper
                upper_t_lim = (Mo**(1-n))/(K*(1-n))


                term3 = K*(n-1)
                term4 = Mo**(1-n)
                term5 = term3*term4*t
                power = round(1/(1-n),10)

                wt = (1- (1+(term5))**power)*wm # calculate wm
            else:

                wt = wm*(1- np.exp(-K*t))
            return wt/12.5

        def objective(params, x_data, y_data):

            model_predictions = your_model_function(params, x_data)
            residuals = model_predictions - y_data
            return np.sum(residuals**2)
        
        def calculate_r2(params, x_data, y_data):
            K,n=params
            upper_t_lim = (1**(1-n))/(K*(1-n))
            
            new_x_data = []
            new_y_data = []
            
            for i in range(len(x_data)):
                if x_data[i]<upper_t_lim:
                    new_x_data.append(x_data[i])
                    new_y_data.append(y_data[i])
                        
            
            model_predictions = your_model_function(params, np.array(new_x_data))
            r2_value = r2_score(np.array(new_y_data), model_predictions)
            return r2_value

        def fit_data(x_data, y_data):
            # Initial guess for the parameters
            initial_guess = [0.15, 0.9]  # Adjust as needed
            bounds = [(0.0001, 0.1), (0.1, 0.99)]  
            # Perform the optimization using minimize
            result = minimize(objective, initial_guess, args=(x_data, y_data),
                              method='L-BFGS-B', bounds=bounds)

            opt_params = result.x
            r2_val = calculate_r2(opt_params, x_data, y_data)

            return result.x[0],result.x[1], r2_val
        
        if k == None:
            optimal_params = fit_data(xdata, ydata)
            #print("K = ", optimal_params[0])
            #print("n = ", optimal_params[1])
            #print("gives R2 = ",optimal_params[2])
        else:
            optimal_params = (k,n)
            #print("K = ", k)
            #print("n = ", n)
            r= calculate_r2((k,n),xdata,ydata)
            #print("gives R2 = ",r)
            optimal_params = (k,n,r)

        return optimal_params
