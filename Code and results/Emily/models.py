import numpy as np

class shrinking_core:

    def visccosity(self, T):
        return (8.76*10e-6)*(365.85/(T+72))*((T/293.85)**1.5)

    def diffusitivity_helium(self, T):
        return 1.032*1e-8 * (T**1.74)

    def reaction_rate(self, T):
        return 0.51*((8.314*T)**0.5)*np.exp(-3033/T)#1#10.4*np.exp(-1592/T)


    def diffusivity_constant(self, T, a):
        return a*np.exp(-2300/(8.314*T))



    def core_radius(self, initial_pellet_size, shrinking_step):
        core_radius_values =  np.arange(initial_pellet_size,-1*shrinking_step , - shrinking_step )
        return core_radius_values

    #r_c = core_radius(initial_pellet_size, shrinking_step)
    def outer_radius(self, initial_pellet_size, rho_u, rho_uh3, r_c, N):
        outer_radius_values = np.zeros(N+1)
        for i in range(N+1):
            outer_radius_values[i] =((((initial_pellet_size*1e-6)**3)-((r_c[i]*1e-6)**3))*
                                     (rho_u/rho_uh3)+((r_c[i]*1e-6)**3))**(1/3)
        return outer_radius_values

    def mass_transfer(self, r_o,d, viscosity, r_c,N, rho_h, velocity):
        kg = np.zeros(N+1)
        sc = viscosity/d
        for i in range(N+1):
            re = rho_h*velocity*2*r_c[i]*1e-6/viscosity
            kg[i] = (2 + (0.6*(sc**(1/3)) * (re**0.5)))*(d)/(2*r_o[i]*1e-6)
        return kg



    def const(self, kr, kg, r_c, r_o, de, N):
        a =  np.zeros(N+1)
        b =  np.zeros(N+1)
        e =  np.zeros(N+1)
        const = np.zeros(N+1)
        for i in range(N+1):
            a[i] = (((r_c[i]*1e-6)**2)*kr)/((r_o[i]**2)*kg[i])
            b[i] = ((r_c[i]*1e-6)*kr)/(de)
            e[i] = (((r_c[i]*1e-6)**2)*kr)/(de*r_o[i])
            const[i] = a[i]+b[i]-e[i]
        return const



    def core_concentration(self, c_b, const,N):
        c_c = np.zeros(N+1)
        for i in range(N+1):
            c_c[i] = ((-const[i] + np.sqrt((const[i]**2)+(4*c_b)))/2)**2
        return c_c

    #c_c = core_concentration(c_b)
    #print(c_c)

    def time_taken(self, b, c_c, shrinking_step, kr, rho_u, N):
        time_diff = np.zeros(N+1)
        for i in range(N+1):
            time_diff[i] = ((shrinking_step* 1e-6)*rho_u)/(b*kr*np.sqrt(c_c[i]))#

        time = np.cumsum(time_diff)
        return time



    def thiele_modulus(self, initial_pellet_size , kr, c_c, de):
        m = 2*initial_pellet_size*1e6 *(np.sqrt((1.5*kr*(c_c**(-0.5)))/2*de))
        return m

    def simulation(self, T, initial_pellet_size, p, N, a,rho_h,rho_u,rho_uh3,velocity, diff=None):
        if diff==None:
            de = self.diffusivity_constant(T, a)
        else:
            de = diff
        shrinking_step = initial_pellet_size/N
        c_b =  p/(8.314*T)
        d = self.diffusitivity_helium(T)
        viscosity = self.visccosity(T)
        kr = self.reaction_rate(T)  
        #de = diffusivity_constant(T, a)
        r_c = self.core_radius(initial_pellet_size, shrinking_step)
        r_o = self.outer_radius(initial_pellet_size, rho_u, rho_uh3, r_c,N)
        kg = self.mass_transfer(r_o,d, viscosity, r_c, N, rho_h,velocity)
        const1 = self.const(kr,kg, r_c, r_o,de,N)
        c_c = self.core_concentration(c_b, const1,N)
        time = self.time_taken(0.667, c_c, shrinking_step, kr,rho_u,N)
        x = r_c/initial_pellet_size
        relitive_sphere_radius = r_o/(initial_pellet_size*1e-6)
        pressure_change = (1800 - ((3/((4/3)*3.14*((initial_pellet_size*1e-6)**3)*19.1e6))*((4/3)*3.14*((initial_pellet_size*1e-6)**3 - (r_c*1e-6)**3))*rho_u*1.5*8.314*T/0.00048)/100)
        m = self.thiele_modulus(initial_pellet_size , kr, c_c, de)
        y =  np.log10(c_c)
        results = [time, x, pressure_change, y, relitive_sphere_radius, m, relitive_sphere_radius, r_o]
        return results

    
class Stakebake():
    def stakebake(K, n):
        wm=12.5
        if n != 1:
            Mo =1
            #calclaute upper limit of t based on stakebake paper
            upper_t_lim = (Mo**(1-n))/(K*(1-n))


            #create array of times between 0 and upper lim in steps of 0.1
            t=np.linspace(0,upper_t_lim, 10001)
            term3 = K*(n-1)
            term4 = Mo**(1-n)
            term5 = term3*term4*t
            power = round(1/(1-n),10)
            
            wt = (1- (1+(term5))**power)*wm # calculate wm
        else:
            t = np.linspace(0,150, 1001)
            #t = t/100
            wt = wm*(1- np.exp(-K*t))
        return wt, t
    
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
    
    
    
    
class BM():
    def Pdcalcpa(self,T):
        term1 = (8.394e11)
        term2 = np.exp(-11200/T)
        return term1*term2

    def Kmcalc(self,T):
        expterm = np.exp(1570/T)
        return (7.99e-5)*expterm

    def bm_ck(self, T, P):

        Km = self.Kmcalc(T)
        Pd = self.Pdcalcpa(T)
        wg = []
        rootPd = np.sqrt(Pd)
        rootP = np.sqrt(P)
        term1 = 358.94
        term2 = np.exp(-3829/T)
        term3 = ((Km*rootPd)/(1+(Km*rootPd)))**2
        term4 = P/Pd
        term5 = ((1+(Km*rootPd))/(1+(Km*rootP)))**2
        term6 = rootPd/rootP
        wg.append(term1*term2*term3*((term4*term5) - term6))
        wg = np.array(wg)
        wg[wg<0]=0.000000000001
        return wg
