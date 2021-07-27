import os, sys
import numpy as np

# import matplotlib.pyplot as plt
#  %matplotlib inline

import scipy 
from scipy import stats	


### Functions ###

def get_coul_vdW_lambdas(mdpfile):
    """Given an *.mdp file as input, extract the values of coul-lambdas and vdw-lambdas
    
    RETURNS
    coul_lambdas    - numpy array of coul-lambdas
    vdw_lambdas     - numpy array of vdw-lambdas
    
    NOTE: for the Moonshot2 alchemical transformations, the lambda=0 state is fully coupled, 
          and lambda=1 is fully uncoupled:
          
              couple-lambda0         = vdw-q
              couple-lambda1         = none
              
          That means that *first* the coulomb gets turned off, *then* the vdW:
          
    coul-lambdas         = 0.0 0.020 0.040 0.061 0.081 ... 0.979 1.0 1.000 1.000 1.000 1.000 ... 1.000 1.0
    vdw-lambdas          = 0.0 0.000 0.000 0.000 0.000 ... 0.000 0.0 0.020 0.040 0.061 0.081 ... 0.979 1.0
    """
        
    fin = open(mdpfile,'r')
    lines = fin.readlines()
    fin.close()
    
    coul_lambdas, vdw_lambdas = None, None
    
    for line in lines:
        if line.count('coul-lambdas'):
            coul_string = line.split('=')[1].strip()
            # print('coul_string', coul_string)
            coul_lambdas = np.array([float(s) for s in coul_string.split()])
            # print('coul_lambdas', coul_lambdas)
        elif line.count('vdw-lambdas'):
            vdw_string = line.split('=')[1].strip()
            # print('vdw_string', vdw_string)
            vdw_lambdas = np.array([float(s) for s in vdw_string.split()])

    return coul_lambdas, vdw_lambdas

def get_dhdl_data(dhdl_xvgfile, verbose=True):
    """Read and parse the information in the dhdl file.
    
    RETURNS
    time_in_ps      - time in ps (1D np.array)
    thermo_states   - thermodynamic state indices (1D np.array)
    dhdl            - delta_Uij (np.array of shape (N,K))
                      where N is snapshots and K is number of thermodynamic states
    
    NOTES
    
    time         ---> Column 0 is time in ps.                         
    thermo_index ---> Column 1 = @ s0 legend "Thermodynamic state"
                      Column 2 = @ s1 legend "Total Energy (kJ/mol)"
                      Column 3 = @ s2 legend "dH/d\\xl\\f{} fep-lambda = 0.0000"
                      Column 4 = @ s3 legend "dH/d\\xl\\f{} coul-lambda = 0.0000"
                      Column 5 = @ s4 legend "dH/d\\xl\\f{} vdw-lambda = 0.0000"
    dU_ij starts      Column 6 = @ s5 legend "\\xD\\f{}H \\xl\\f{} to (0.0000, 0.0000, 0.0000)"
                      Column 7 = @ s6 legend "\\xD\\f{}H \\xl\f{} to (0.0000, 0.0200, 0.0000)"
                      Column 8 = @ s7 legend "\\xD\\f{}H \\xl\\f{} to (0.0000, 0.0400, 0.0000)"
    """
    
    
    import os
    assert os.path.exists(dhdl_xvgfile)
    
    # Read and parse the file
    fin = open(dhdl_xvgfile,'r')
    lines = fin.readlines()
    fin.close()
    
    dhdl_column_start = None
    # Read to headers to find which column starts the dhdl data
    for line in lines:
        # Looking for line like this: "@ s5 legend "\xD\f{}H \xl\f{} to (0.0000, 0.0000, 0.0000)"
        if line.count('"\\xD\\f{}H') > 0:
            dhdl_column_start = int(line.split(' ')[1].replace('s','')) + 1
            break
    if verbose:
        print('dhdl data starts at column:', dhdl_column_start)

    # Get rid of all the header lines
    i = 0
    while i < len(lines):
        if (lines[i][0] == '#') or (lines[i][0] == '@'):
            lines.pop(i)
        else:
            i+=1
    
    # find the correct number of entries from the first line
    ncols = len(lines[0].strip().split())
    print(lines[0])
    print('ncols', ncols)
    
    time_in_ps, dhdl, thermo_states = [], [], []
    for line in lines:
        line_data_list = [float(s) for s in line.strip().split()] 
            
        # Skip line if it doesn't have the correct number of entries
        # (sometimes the I/O gets cutoff when writing the dhdl.xvg in it corrupts the data)
        if len(line_data_list) == ncols:
            time_in_ps.append(line_data_list[0])
            thermo_states.append(line_data_list[1])
            dhdl.append(line_data_list[dhdl_column_start:])
            
    time_in_ps = np.array(time_in_ps)
    dhdl = np.array(dhdl)
    thermo_states = np.array(thermo_states)
    
    print('time_in_ps.shape', time_in_ps.shape)
    print('dhdl.shape=', dhdl.shape)
    print('thermo_states=', thermo_states)

    return time_in_ps, thermo_states, dhdl




def estimate_sigmas(dhdl, thermo_states, plot_data=True):
    """Using as input the Delta_U_ij energies from the dhdl array, 
    estimate the standard deviations P(U_{i-->i+1}) for neighboring ensembles.
    
    RETURNS
    sigmas   - a np.array() of standard deviations P(U_{i-->i+1}).
    """
    
    nlambdas = dhdl.shape[1]
    
    if plot_data:
        plt.figure(figsize=(6, 80))

    Delta_uij_values = []     
    sigmas = []
       
    for j in range(nlambdas-1):
    
        ##transitions from state 0 to 1 or 1 to 2, or 2 to 3 .... 
        Ind_i = (thermo_states == j)
        delta_u_ij = dhdl[Ind_i, j+1]       ##only for neighbored ensembles

        #Delta_uij_values.append(delta_u_ij)

        mu, sigma = scipy.stats.norm.fit(delta_u_ij)
        #print (mu, sigma)
        sigmas.append(sigma)

        delta_u_bins = np.arange(-15., 15., 0.2)
        counts, bin_edges = np.histogram(delta_u_ij, bins=delta_u_bins)
        counts = counts/counts.sum() # normalize
        bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2.0

        if plot_data:
            plt.subplot(nlambdas-1, 1, j+1)
            plt.step(bin_centers, counts, label='$\Delta u_{%d \\rightarrow %d} \sigma$=%.2f'%(j,j+1,sigma))
            #plt.xlabel('$\Delta u_{%d \\rightarrow %d}$'%(j, j+1))
            plt.legend(loc='best')
        
    if plot_data:
        plt.tight_layout()
        plt.show()

    return np.array(sigmas)



###################################





### Main ###

if __name__ == '__main__':

    usage  = """Usage:    python optimize_coul_vdw_lambdas.py mdpfile dhdl_xvgfile

    DESCRIPTION
        This script will optimize the lambda values for coul-then-vdw decoupling intermediates
        to minimize the total variance in P(\Delta u_ij) for neighboring thermodynamic ensembles

    OUTPUT
        A mdpfile-compatible string with new coul-lambdas and vdw-lambdas will be printed to std output

    EXAMPLE

    Try this:
        $ python optimize_coul_vdw_lambdas.py ../x11294_L/RUN0/prod.mdp ../x11294_L/RUN0/dhdl.xvg

    """        

    # Parse input

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    mdpfile      = sys.argv[1]
    dhdl_xvgfile = sys.argv[2]
    
    coul_lambdas, vdw_lambdas = get_coul_vdW_lambdas(mdpfile)

    # We map each [0,1] set of values to the interval [0,2] by elemenet-wise summing of the two sets of values
    lambdas = coul_lambdas + vdw_lambdas
    print('lambdas', lambdas)
    print('lambdas.shape', lambdas.shape)

 
    time_in_ps, thermo_states, dhdl = get_dhdl_data(dhdl_xvgfile)
    print('time_in_ps', time_in_ps)
    print('thermo_states', thermo_states)
    print('dhdl', dhdl)

    sigmas = estimate_sigmas(dhdl, thermo_states, plot_data=False)


    ### HOT FIX
    ### If any of the sigma values are zero, then we know this is one of the cases where there are duplicate
    ### coul-lambda and vdw-lambda

    print('sigmas', sigmas)
    indices_for_which_sigma_is_zero = np.where(sigmas < 0.0001)[0]
    print('indices_for_which_sigma_is_zero', indices_for_which_sigma_is_zero)

    # There should be only one such duplicate!
    if len(indices_for_which_sigma_is_zero) == 1:
        # For *THIS* particular data set we can see there is a problem:
        # The i=49 --> i+1 = 50 lambdas are the same, resulting in a sigma of zero for that transition
        # Let's remove it
     
        remove_index = indices_for_which_sigma_is_zero[0]
        print('Removing DUPLICATE lambda index:', remove_index)
        # print('sigmas', sigmas)
        sigmas_list = sigmas.tolist()
        sigmas_list.pop(remove_index)
        sigmas = np.array(sigmas_list)
        print('FIXED sigmas', sigmas)

        print('lambdas',lambdas)
        print('lambdas.shape',lambdas.shape)
        lambdas_list = lambdas.tolist()
        lambdas_list.pop(remove_index)
        lambdas = np.array(lambdas_list)
        print('FIXED lambdas',lambdas)
        print('FIXED lambdas.shape',lambdas.shape)

    elif len(indices_for_which_sigma_is_zero) > 1:
        raise Exception("There are multiple zero sigma! Are there multiple duplicates in the lambda values?!")

    else:
        print('There are no duplicate lambda')
        pass


    ### Lambda optimization

    dx = sigmas                 #according to Vince's equation (VAV: k is set to 1)
    
    x_values = np.cumsum(dx)    # convert to a list of x values of separated harmonic potentials
    x_values = np.array(np.concatenate([[0], x_values]))    # add a zero corresponding to lambda0 = 0.0
    ## VAV: This zero needs to be included.  Why was this left out before?
    print('x_values', x_values)


    from scipy.interpolate import UnivariateSpline
    from scipy.interpolate import interp1d

    make_plots = False

    if make_plots:
        plt.figure(figsize=(12,6))

    lambda_values = lambdas #not inclduing the first one, lambda_0 

    x_observed = lambda_values      #not inclduing the first one, lambda_0
    y_observed = x_values

    if make_plots:
        plt.subplot(1,2,1)
        plt.plot(x_observed, y_observed, 'ro', label = 'data')
        #plt.semilogy(x_observed, y_observed, 'ro', label = 'data')

    #y_spl = CubicSpline(x_observed, y_observed)#, s=0,k=4)  
    y_spl = UnivariateSpline(x_observed, y_observed, s=0, k=3)  
    x_range = np.linspace(x_observed[0], x_observed[-1], 1000)


    if make_plots:
        plt.plot(x_range, y_spl(x_range), label="spline")   # for UnivariateSpline
        ## plt.plot(x_observed, y_spl(x_observed), label="spline") # for CubicSpline
        plt.legend()
        plt.xlabel('lambda')
        plt.ylabel('x values')

        plt.subplot(1,2, 2)   #derivative plot

    y_spl_1d = y_spl.derivative(n=1)    #n=1 , means the first order derivative
    #print (y_spl_1d(x_observed))
    # y_spl_1d = y_spl(x_observed, 1)  # first derivative of Cubic spline

    if make_plots:
        plt.plot(x_range, y_spl_1d(x_range), '-')
        plt.plot(x_observed, y_spl_1d(x_observed), '.')
        plt.ylabel('dx/dlambda')
        
        #plt.plot(x_observed, y_spl_1d, '.-', label='derivative')
        plt.legend()
        plt.xlabel('lambda')


    # Let's try a steepest descent algorithm like the kind I wrote up in "math-gradient-descent-2021-05-07.pdf"

    # run the algorithm some fixed number of steps, or until some tolerance is reached
    nsteps = 100000
    tol = 1e-7  # stop if the lambdas dont change within this tolerance

    alpha = 1e-5  # gradient descent step size
    max_del_lambda = 0.0001   # the minimization step limited to this as a maximum change

    VERBOSE = False
    print_every = 250
    
    nlambdas = len(lambda_values)
    print('lambda_values', lambda_values)
    old_lambdas = np.array(lambda_values)
    traj_lambdas = np.zeros( (nlambdas,nsteps) )
    for step in range(nsteps):

        # store the trajectory of lambdas
        traj_lambdas[:,step] = old_lambdas
        if VERBOSE:
            print('step', step, old_lambdas)
    
        # perform a steepest descent step
        new_lambdas = np.zeros( old_lambdas.shape )
        del_lambdas = np.zeros( old_lambdas.shape )
        del_lambdas[0] = 0.0   # fix the \lambda = 0 endpoint
        del_lambdas[nlambdas-1] = 0.0  # fix the \lambda = 1 endpoint
    
        if False:  # do in a loop (SLOW!) 
            for i in range(1, (nlambdas-1)):
                del_lambdas[i] = -1.0*alpha*2.0*y_spl_1d(old_lambdas[i])*( 2.0*y_spl(old_lambdas[i]) - y_spl(old_lambdas[i-1]) - y_spl(old_lambdas[i+1]))
        else:   # do as a vector operation (FAST!) 
            y_all = y_spl(old_lambdas)
            yh, yi, yj = y_all[0:nlambdas-2], y_all[1:nlambdas-1], y_all[2:nlambdas] 
            del_lambdas[1:nlambdas-1] = -1.0*alpha*2.0*y_spl_1d(old_lambdas[1:nlambdas-1])*( 2.0*yi - yh - yj)
        if abs(np.max(del_lambdas)) > max_del_lambda:
            del_lambdas[1:nlambdas-1] = del_lambdas[1:nlambdas-1]*max_del_lambda/np.max(del_lambdas)
        new_lambdas = old_lambdas + del_lambdas
        
        # record the average change in the lambdas 
        del_lambdas = np.abs(old_lambdas - new_lambdas).mean()
        if step % print_every == 0:
            print('step', step, 'del_lambdas', del_lambdas)
        if del_lambdas < tol:
            print('Tolerance has been reached: del_lambdas =', del_lambdas, '< tol =', tol)
            break
        
        old_lambdas = new_lambdas
   
    if make_plots:     
    
        # Plot the results
        plt.figure(figsize=(12,4))

        plt.subplot(1,2,1)
        for i in range(nlambdas):
            plt.plot(range(step), traj_lambdas[i,0:step], '-')
        plt.xlabel('step')
        plt.ylabel('lambda values')
        
        plt.subplot(1,2,2)
        for i in range(nlambdas):
            plt.plot(range(step), y_spl(traj_lambdas[i,0:step]), '-')
        plt.xlabel('step')
        plt.ylabel('x values')


    if make_plots:

        plt.figure(figsize=(12,4))

        plt.subplot(2,1,1)
        plt.plot(x_range, y_spl(x_range), 'b-', label="spline")
        plt.plot(lambda_values, y_spl(np.array(lambda_values)), 'r.', label="old lambdas")
        for value in lambda_values:
            plt.plot([value, value], [0, y_spl(value)], 'r-')
        plt.legend()
        plt.xlabel('lambda')
        plt.ylabel('x values')
        plt.title('old lambdas')

        plt.subplot(2,1,2)
        plt.plot(x_range, y_spl(x_range), 'b-', label="spline")
        plt.plot(new_lambdas, y_spl(new_lambdas), 'g.', label="new lambdas")
        for value in new_lambdas:
            plt.plot([value, value], [0, y_spl(value)], 'g-')
        plt.legend()
        plt.xlabel('lambda')
        plt.ylabel('x values')
        plt.title('new lambdas')


    # Finally, we transform the [0,2] coul+vdW interval back to separate coul_lambdas and vdw_lambdas
    new_coul_lambdas = np.minimum(new_lambdas, np.ones(new_lambdas.shape))
    new_vdw_lambdas  = np.maximum(new_lambdas, np.ones(new_lambdas.shape)) - 1.0

    #print('new_coul_lambdas',new_coul_lambdas)
    #print('new_vdw_lambdas',new_vdw_lambdas)
    
    # print out the new lambdas as if they were in an mdp file
    outstring = 'coul-lambdas    = ' + " ".join(['%1.4f'%lam for lam in new_coul_lambdas])
    print(outstring)
    
    outstring = 'vdw-lambdas     = ' + " ".join(['%1.4f'%lam for lam in new_vdw_lambdas])
    print(outstring)

