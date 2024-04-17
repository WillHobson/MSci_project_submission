"""In this python script are the functions needed to run the time
dependent Bloch-Mintz model"""


"""
dependencies
"""
import numpy as np


class BM():
    """
    This function uses the Condon equation to calculate the equilibrium pressure
    at a given temperature. 
    INPUT: temperature( K)
    OUTPUT: pressure (Pa)     
    """
    def Pdcalcpa(self, T):
        term1 = (8.394e11)
        term2 = np.exp(-11200/T)
        return term1*term2

    """
    This function is the original bm equation which calculates the growth rate as a velocity
    INPUT: temperature (K)
           pressure (Pa)
    OUTPUT: growth rate (um/s)     
    """

    def growthrate(self, T, P):
        n=2
        Pd = self.Pdcalcpa(T)
        Pd_torr = Pd*0.00750062
        Km_squared = (1/((self.Kmcalc(T))**2))*0.00750062
        Km_torr = 1/np.sqrt(Km_squared)
        Keq = self.KeqmCK(T, self.Kmcalc(T), Pd)
        rootPd = Pd**(1/2)
        rootP = P**(1/2)
        term2 = (P/Pd)**(n/2)
        term3 = ((1+(Km*rootPd))/(1+(Km*rootP)))**n
        Ug = Keq*((term2 * term3) - (rootPd/rootP))
        return Ug

    """
    Value for equilibrium constant in growth rate using Loui's Km equation
    """
    def KeqmCK(self, T, km,peq):
        term1 = np.exp(-3829/T)
        term2 = km*np.sqrt(peq)/(1+(km*np.sqrt(peq)))
        return 358.94*term1*(term2**2)

    """
    Kmcalc gives the value of Km in the BM equation. This is as presented in a revision
    to the model by Loui.
    INPUT: temperature (K)
    OUTPUT: Km (sqrt[1/Pa])
    """
    def Kmcalc(self, T):
        expterm = np.exp(1570/T)
        return (7.99e-5)*expterm


    """
    bm_ck is an implementation of the revised BM model given by Loui. This is an adaptation of
    the original equation in the BM paper for growth rate. This gives the weight gain rate of the
    uranium sample
    INPUT: temperature (K)
           pressure (Pa)
           activation energy of hydride formation (kj/mol)
           proportionality constant for activation energy as function of reaction progression (kj/mol)
               *set to 0 for constant qf
           reaction progression (dimensionless)
    OUTPUT: weight gain rate (molH/m**2 U /s)
    """

    def bm_ck(self, T, P, qf, A,L):
        Km = self.Kmcalc(T)
        Pd = self.Pdcalcpa(T)
        wg = []
        rootPd = np.sqrt(Pd)
        rootP = np.sqrt(P)
        term1 = 358.94

        #term2 = np.exp(-(qf+(A*L))/(8.31*T))
        eaterm = -qf-(A*L)

        term2 = np.exp(eaterm/(8.31*T))
        term3 = ((Km*rootPd)/(1+(Km*rootPd)))**2
        term4 = P/Pd
        term5 = ((1+(Km*rootPd))/(1+(Km*rootP)))**2
        term6 = rootPd/rootP
        wg.append(term1*term2*term3*((term4*term5) - term6))
        wg = np.array(wg)
        wg[wg<0]=0.000000000001
        return wg

    """
    simulation_const_p simulates the hydriding reaction under isothermic and isobaric conditions
    This is the case for experiments which replace hydrogen gas as it is consumed by the uranium
    INPUT: initial pressure (Pa)
           temperature (K)
           mass of uranium (g)
           volume of gas rig (m**3)
           initial radius of particle
           time step (0.1 recommended)
           activation energy for hydride formation (kj/mol)
           proportionality constant for activation energy as function of reaction progression (kj/mol)
           geometric factor given by the ratio of surface area to volume (3 recommend for sphere)
    OUTPUT: t_array = time data in seconds
            mol array = measure of reaction progression, denoted alpha in report
            radii array = decreasing radii values as the uranium core contracts
            SA array = surface area values
    """

    def simulation_const_p(self, P_init, T, M_U, V, Rad_init, t_step, qf, A, shape=None):
        R = 8.31                   #Ideal gas constant
        mol_U = M_U/238            #Moles Uranium
        rho_U = 19.1e3             #Density Uranium
        imHsys = P_init*V/(R*T)    #initial moles hydrogen in system
        Hmsys = imHsys             #moles of H in system

        if shape == 'sphere':
            g=3
        elif shape == 'cylinder':
            g=2
        elif shape == None:
            g=3
        else:
            g=3


        SA_init = g*((M_U/1000)/rho_U)/Rad_init    #surface area of particle

        MTmolG = 0.0125*M_U/2      #max theoretical absorbed H mols
        MTmolG  = 1.5*mol_U       #max theoretical absorbed H mols


        SA_array = [SA_init]       #array to track surface area
        t_array = [0]              #array to track time
        mol_array = [0]            #array to track mols absorbed
        tot_mol_abs = 0            #variable to track progression
        radii_array = [Rad_init]   #array to track the particle radius over time
        r_hyd = [0]


        wgs = [self.bm_ck(T,P_init,qf, A, r_hyd[-1])]  #array to track weight gain rates

        #while reaction is still ongoing
        while tot_mol_abs/MTmolG < 0.999999:
            P = P_init                           #take const pressure
            wg = self.bm_ck(T, P,qf, A, r_hyd[-1])           #calculate the weight gain at that pressure
            mol_abs = (wg[0]*SA_array[-1]*t_step)#moles absorbed in one time step
            tot_mol_abs += mol_abs               #add moles absorbed to total mols absorbed
            mol_array.append(tot_mol_abs/MTmolG)
            Hmsys = Hmsys - mol_abs              #calculate mols H left in system
            SA_current = SA_init*(1-(tot_mol_abs/MTmolG))    #change in SA
            r_current = Rad_init*(1-(tot_mol_abs/MTmolG))
            radii_array.append(r_current)
            t_array.append(t_array[-1]+t_step)   #record the time it was here
            wgs.append(wg)                       #record the weight gain rate
            SA_array.append(SA_current)          #record the SA
            r_hyd.append(tot_mol_abs/MTmolG)
            if len(mol_array)>100000:            #include a break in case simulation doesnt reach completion
                tot_mol_abs = MTmolG

        return t_array, mol_array, radii_array, SA_array


    """
    simulation_ simulates the hydriding reaction under isothermic conditions but for experiments which 
    the consumed hydrogen is not replaced and the pressure falls. In this case the system pressure must
    be updated at every time step

    INPUT: initial pressure (Pa)
           temperature (K)
           mass of uranium (g)
           volume of gas rig (m**3)
           initial radius of particle
           time step (0.1 recommended)
           activation energy for hydride formation (kj/mol)
           proportionality constant for activation energy as function of reaction progression (kj/mol)
           geometric factor given by the ratio of surface area to volume (3 recommend for sphere)
    OUTPUT: t_array = time data in seconds
            mol array = measure of reaction progression, denoted alpha in report
            radii array = decreasing radii values as the uranium core contracts
            SA array = surface area values
    """
    def simulation(self, P_init, T, M_U, V, Rad_init, t_step, qf,A, shape=None):
        #Initial parameters
        R = 8.31                   #Ideal gas constant
        mol_U = M_U/238            #Moles Uranium
        rho_U = 19.1e3             #Density Uranium
        imHsys = P_init*V/(R*T)    #initial moles hydrogen in system
        Hmsys = imHsys             #moles of H in system

        if shape == 'sphere':
            g=3
        elif shape == 'cylinder':
            g=2
        elif shape == None:
            g=3
        else:
            g=3

        SA_init = g*((M_U/1000)/rho_U)/Rad_init    #surface area of particle

        MTmolG = 0.0125*M_U/2      #max theoretical absorbed H mols
        MTmolG  = 1.5*mol_U       #max theoretical absorbed H mols

        P_end = P_init - (MTmolG*R*T/V)    #Final pressure when MTmolG complete

        P_array = [P_init]         #array to track pressure
        SA_array = [SA_init]       #array to track surface area
        t_array = [0] #array to track time
        r_hyd = [0]
        radii_array = [Rad_init]
        tot_mol_abs = 0            #variable to track progression
        mol_array = [0]            #array to track mols absorbed



        wgs = [self.bm_ck(T,P_init,qf, A, r_hyd[-1])]  #array to track weight gain rates

        #while reaction is still ongoing
        while tot_mol_abs/MTmolG < 0.999999:
            P = P_array[-1]                      #take latest pressure
            wg = self.bm_ck(T, P,qf, A, r_hyd[-1])           #calculate the weight gain at that pressure
            mol_abs = (wg[0]*SA_array[-1]*t_step)#moles absorbed in one time step
            tot_mol_abs += mol_abs               #add moles absorbed to total mols absorbed
            mol_array.append(tot_mol_abs/MTmolG)

            Hmsys = Hmsys - mol_abs      #calculate mols H left in system
            SA_current = SA_init*(1-(tot_mol_abs/MTmolG))    #change in SA
            P_change = P-(mol_abs*R*T/V)         #pressure change from absorbed moles H
            P_array.append(P_change)             #record new pressure
            t_array.append(t_array[-1]+t_step)   #record the time it was here
            wgs.append(wg)                       #record the weight gain rate
            SA_array.append(SA_current)
            r_current = Rad_init*(1-(tot_mol_abs/MTmolG))#record the SA
            radii_array.append(r_current)
            #r_hyd.append(Rad_init-r_current)
            r_hyd.append(tot_mol_abs/MTmolG)

            if len(mol_array)>100000:
                tot_mol_abs = MTmolG

        results = [t_array, P_array, wgs, SA_array,radii_array,r_hyd, mol_array]
        return results
