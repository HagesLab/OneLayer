

import numpy as np
from scipy import integrate as intg
from utils import to_array
import tables

from Modules.module_MAPI_Rubrene_DBP.simulations_dydt import dydt_sct
from Modules.module_MAPI_Rubrene_DBP.simulations_dydt import dydt_basic
from Modules.module_MAPI_Rubrene_DBP.calculations import E_field
from Modules.module_MAPI_Rubrene_DBP.calculations import SST


def ode_twolayer(data_path_name, m, dm, f, df, n, dt, mapi_params, ru_params, 
                 do_Fret=False, do_ss=False, no_upconverter=False, predict_sstriplets=False,
                 do_seq_charge_transfer=False,
                 hmax_=0, write_output=True, 
                 init_N=0, init_P=0, init_T=0, init_S=0, init_D=0, init_P_up=0):
    """
    Master function for MAPI_Rubrene_DBP module simulation.
    Problem statement:
    Create a discretized, time and space dependent solution (N(x,t), P(x,t), T(x,t), S(x,t), D(x,t))
    of a MAPI-Rubrene:DBP carrier model with m, f space steps and n time steps
    Space step size is dm, df; time step is dt
    Initial conditions: init_N, init_P, init_T, init_S, init_D
    Optional FRET integral term

    Parameters
    ----------
    data_path_name : str
        Output file location.
    m : int
        Number of MAPI layer space nodes.
    dm : float
        MAPI Space node width.
    f : int
        Number of Rubrene layer space nodes.
    df : float
        Rubrene Space node width.
    n : int
        Number of time steps.
    dt : float
        Time step size.
    mapi_params : dict {"str":Parameter}
        Collection of parameter objects for MAPI layer
    ru_params : dict {"str":Parameter}
        Collection of parameter objects for Rubrene layer
    do_Fret : bool, optional
        Whether to include the FRET integral. The default is True.
    do_ss : bool, optional
        Whether to inject the initial conditions at every time step, creating a nonzero steady state situation. The default is False.
    no_upconverter : bool, optional
        Whether to block new triplets from being formed at the MAPI/Rubrene interface, which effectively deactivates the latter upconverter layer.
        The default is False.
    predict_sstriplets : bool, optional
        Whether to start the triplet density at a predicted steady state value rather than zero.
        This reduces the simulation time needed to reach steady state.
        This only works if do_ss is also active.
        The default is False.
    hmax_ : float, optional
        Maximum internal step size to be taken by ODEINT. The default is 0.
    write_output : bool, optional
        Whether to write output files. TEDs always does this but other applications reusing this function might not. The default is True.
    init_N : 1D ndarray, optional
        Initial excited electron distribution. The default is 0.
    init_P : 1D ndarray, optional
        Initial hole distribution. The default is 0.

    Returns
    -------
    None
        TEDs does not do anything with the return value. Other applications might find this useful however.
    """
    
    ## Set data type of array files
    atom = tables.Float64Atom()

    ## Unpack params; typecast non-array params to arrays if needed
    Sf = mapi_params["Sf"].value
    Sb = mapi_params["Sb"].value
    if do_seq_charge_transfer:
        # Unpack additional params for this physics model
        Sp = 0 if no_upconverter else ru_params["Sp"].value
        Ssct = ru_params["Ssct"].value
        w_vb = ru_params["W_VB"].value
        mu_p_up = to_array(ru_params["mu_P_up"].value, f, True)
        uc_eps = to_array(ru_params["uc_permitivity"].value, f, True)

    else:
        St = 0 if no_upconverter else ru_params["St"].value

    mu_n = to_array(mapi_params["mu_N"].value, m, True)
    mu_p = to_array(mapi_params["mu_P"].value, m, True)
    mu_s = to_array(ru_params["mu_S"].value, f, True)
    mu_T = to_array(ru_params["mu_T"].value, f, True)
    mapi_temperature = to_array(mapi_params["MAPI_temperature"].value, m, True)
    rubrene_temperature = to_array(ru_params["Rubrene_temperature"].value, f, True)
    n0 = to_array(mapi_params["N0"].value, m, False)
    p0 = to_array(mapi_params["P0"].value, m, False)
    T0 = to_array(ru_params["T0"].value, f, False)
    tauN = to_array(mapi_params["tau_N"].value, m, False)
    tauP = to_array(mapi_params["tau_P"].value, m, False)
    tauT = to_array(ru_params["tau_T"].value, f, False)
    tauS = to_array(ru_params["tau_S"].value, f, False)
    tauD = to_array(ru_params["tau_D"].value, f, False)
    B = to_array(mapi_params["B"].value, m, False)
    Cn = to_array(mapi_params["Cn"].value, m, False)
    Cp = to_array(mapi_params["Cp"].value, m, False)
    k_fusion = to_array(ru_params["k_fusion"].value, f, False)
    k_0 = to_array(ru_params["k_0"].value, f, False)
    eps = to_array(mapi_params["rel_permitivity"].value, m, True)
    
   
    ## Package initial condition
    # An unfortunate workaround - create temporary dictionaries out of necessary values to match the call signature of E_field()
    init_E_field = E_field({"N":init_N, "P":init_P}, 
                           {"rel_permitivity":eps, "N0":n0, "P0":p0, "Node_width":dm})
    
    init_E_upc = np.zeros(f+1)
    #init_E_field = np.zeros(m+1)
    
    ## Generate a weight distribution needed for FRET term
    if do_Fret:
        init_m = np.linspace(dm / 2, m*dm - dm / 2, m)
        init_f = np.linspace(df / 2, f*df - df / 2, f)
        weight1 = np.array([1 / ((init_f + (m*dm - i)) ** 3) for i in init_m])
        weight2 = np.array([1 / ((i + (m*dm - init_m)) ** 3) for i in init_f])
        # It turns out that weight2 contains ALL of the parts of the FRET integral that depend
        # on the variable of integration, so we do that integral right away.
        weight2 = intg.trapz(weight2, dx=dm, axis=1)

    else:
        weight1 = 0
        weight2 = 0
    
    if do_ss:
        init_dN = init_N - n0
        init_dP = init_P - p0
        
    else:
        init_dN = 0
        init_dP = 0
        
    if do_ss and predict_sstriplets:
        print("Overriding init_T")
        try:
            np.testing.assert_almost_equal(init_dN, init_dP)
        except AssertionError:
            print("Warning: ss triplet prediction assumes equal excitation of holes and electrons. Unequal excitation is WIP.")
            
        if do_Fret:
            tauD_eff = (k_0*weight2/tauD) + (1/tauD)
        else:
            tauD_eff = 1 / tauD
        init_T, init_S, init_D = SST(tauN[-1], tauP[-1], n0[-1], p0[-1], B[-1], 
                                     St, k_fusion, tauT, tauS, tauD_eff, f*df, np.mean(init_dN))
    
    
    if do_seq_charge_transfer:
        init_condition = np.concatenate([init_N, init_P, init_E_field, init_T, init_S, init_D, init_P_up, init_E_upc], axis=None)
    else:
        init_condition = np.concatenate([init_N, init_P, init_E_field, init_T, init_S, init_D], axis=None)

    

    ## Do n time steps
    tSteps = np.linspace(0, n*dt, n+1)
    
    
    if do_seq_charge_transfer:
        args=(m, f, dm, df, Cn, Cp, 
                tauN, tauP, tauT, tauS, tauD, 
                mu_n, mu_p, mu_s, mu_T,
                n0, p0, T0, Sf, Sb, B, k_fusion, k_0, mapi_temperature, rubrene_temperature,
                eps, uc_eps,
                mu_p_up, Ssct, Sp, w_vb, 
                weight1, weight2, do_Fret, do_ss, 
                init_dN, init_dP)
        sol = intg.solve_ivp(dydt_sct, [0,n*dt], init_condition, args=args, t_eval=tSteps, method='BDF', max_step=hmax_)   #  Variable dt explicit
    else:
        args=(m, f, dm, df, Cn, Cp, 
                tauN, tauP, tauT, tauS, tauD, 
                mu_n, mu_p, mu_s, mu_T,
                n0, p0, T0, Sf, Sb, St, B, k_fusion, k_0, mapi_temperature, rubrene_temperature,
                eps, weight1, weight2, do_Fret, do_ss, 
                init_dN, init_dP)
        sol = intg.solve_ivp(dydt_basic, [0,n*dt], init_condition, args=args, t_eval=tSteps, method='BDF', max_step=hmax_)   #  Variable dt explicit
    
    data = sol.y.T
    e1 = data[:, 2*m:3*m+1]
    e2 = data[:,3*m+1+4*f:]
    if write_output:
        ## Prep output files
        # TODO: Py 3.9 removes need for \
        with tables.open_file(data_path_name + "-N.h5", mode='a') as ofstream_N, \
                tables.open_file(data_path_name + "-P.h5", mode='a') as ofstream_P,\
                tables.open_file(data_path_name + "-T.h5", mode='a') as ofstream_T,\
                tables.open_file(data_path_name + "-delta_S.h5", mode='a') as ofstream_S,\
                tables.open_file(data_path_name + "-delta_D.h5", mode='a') as ofstream_D,\
                tables.open_file(data_path_name + "-P_up.h5", mode='a') as ofstream_P_up:
            array_N = ofstream_N.root.data
            array_P = ofstream_P.root.data
            array_T = ofstream_T.root.data
            array_S = ofstream_S.root.data
            array_D = ofstream_D.root.data
            array_P_up = ofstream_P_up.root.data
            
            array_N.append(data[1:,0:m])
            array_P.append(data[1:,m:2*(m)])
            array_T.append(data[1:,3*(m)+1:3*(m)+1+f])
            array_S.append(data[1:,3*(m)+1+f:3*(m)+1+2*(f)])
            # array_D.append(data[1:,3*(m)+1+2*(f):])
            array_D.append(data[1:,3*(m)+1+2*(f):3*(m)+1+3*(f)])
            if do_seq_charge_transfer:
                array_P_up.append(data[1:, 3*(m)+1+3*(f):3*m+1 + 4*f])
            else:
                # No hole transfer - Insert dummy zeros
                array_P_up.append(np.zeros_like(data[1:,3*(m)+1:3*(m)+1+f:3*m+1 + 4*f]))

        return #error_data

    else:
        array_N = data[:,0:m]
        array_P = data[:,m:2*(m)]
        array_T = data[1:,3*(m)+1:3*(m)+1+f]
        array_S = data[1:,3*(m)+1+f:3*(m)+1+2*(f)]
        # array_D = data[1:,3*(m)+1+2*(f):]
        array_D = data[1:,3*(m)+1+2*(f):3*(m)+1+3*(f)]

        return #array_N, array_P, error_data
