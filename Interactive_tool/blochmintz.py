import numpy as np
from numba import jit

@jit
def Pdcalcpa(T):
    term1 = (8.394e11)
    term2 = np.exp(-11200/T)
    return term1*term2
@jit
def Kmcalc(T):
    expterm = np.exp(1570/T)
    return (7.99e-5)*expterm
@jit
def bm_ck(T, P, qf, A,L):
    Km = Kmcalc(T)
    Pd = Pdcalcpa(T)
    wg = []
    rootPd = np.sqrt(Pd)
    rootP = np.sqrt(P)
    term1 = 358.94
    term2 = np.exp(-(qf)/(8.31*T))
    term3 = ((Km*rootPd)/(1+(Km*rootPd)))**2
    term4 = P/Pd
    term5 = ((1+(Km*rootPd))/(1+(Km*rootP)))**2
    term6 = rootPd/rootP
    wg.append(term1*term2*term3*((term4*term5) - term6))
    wg = np.array(wg)
    wg[wg<0]=0.000000000001
    return wg
@jit
def simulation_const_p(P_init, T, M_U, V, Rad_init, t_step, qf, A):    
    R = 8.31                   #Ideal gas constant
    M_U = 0.9                  #Mass uranium
    mol_U = M_U/238            #Moles Uranium
    rho_U = 19.1e3             #Density Uranium
    imHsys = P_init*V/(R*T)    #initial moles hydrogen in system
    Hmsys = imHsys             #moles of H in system

    SA_init = 3*((M_U/1000)/rho_U)/Rad_init    #surface area of particle

    MTmolG = 0.0125*M_U/2      #max theoretical absorbed H mols
    MTmolG  = 1.5*mol_U       #max theoretical absorbed H mols

    SA_array = [SA_init]       #array to track surface area
    t_array = [0]              #array to track time
    mol_array = [0]            #array to track mols absorbed
    tot_mol_abs = 0            #variable to track progression
    radii_array = [Rad_init]
    r_hyd = [0]
    wgs = [bm_ck(T,P_init,qf, A, r_hyd[-1])]  #array to track weight gain rates

    #while reaction is still ongoing
    while tot_mol_abs/MTmolG < 0.999:

        P = P_init                           #take const pressure
        wg = bm_ck(T, P,qf, A, r_hyd[-1])           #calculate the weight gain at that pressure
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
        if len(mol_array)>10000:
            tot_mol_abs = MTmolG
        r_hyd.append(Rad_init-r_current)
        
    #r_hyd = Rad_init - np.array(radii_array)    
    return t_array, mol_array, radii_array, SA_array, r_hyd


def simulation(P_init, T, M_U, V, Rad_init, t_step, qf,A):
    #Initial parameters
    R = 8.31                   #Ideal gas constant
    mol_U = M_U/238            #Moles Uranium
    rho_U = 19.1e3             #Density Uranium
    imHsys = P_init*V/(R*T)    #initial moles hydrogen in system
    Hmsys = imHsys             #moles of H in system


    SA_init = 3*((M_U/1000)/rho_U)/Rad_init    #surface area of particle

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


    wgs = [bm_ck(T,P_init,qf, A, r_hyd[-1])]  #array to track weight gain rates

    #while reaction is still ongoing
    while tot_mol_abs/MTmolG < 0.99:
        P = P_array[-1]                      #take latest pressure
        wg = bm_ck(T, P,qf, A, r_hyd[-1])           #calculate the weight gain at that pressure
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

        if len(mol_array)>10000:
            tot_mol_abs = MTmolG
    results = [t_array, mol_array, wgs, SA_array,radii_array,r_hyd, mol_array]
    return results