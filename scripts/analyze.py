import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from matplotlib import gridspec
import numpy as np
import pandas as pd
import mdtraj as md
import subprocess, collections
import glob, tqdm, os, sys
import xvg_tools
import harmonic_analytical
import pymbar

dataset = sys.argv[1]
prefix = '..'
data = []

verbose = False
debug = False


for run in range(len(glob.glob(f'{prefix}/{dataset}_RL/RUN*'))):
    if verbose:
        print('run', run)
    scraped_RL_feb_path = f'{prefix}/results/{dataset}/scraped_data/{dataset}_RL_RUN{run}_feb.npy'
    scraped_L_feb_path = f'{prefix}/results/{dataset}/scraped_data/{dataset}_L_RUN{run}_feb.npy'
    if verbose:
        print('Looking for scraped_RL_feb_path', scraped_RL_feb_path, os.path.exists(scraped_RL_feb_path))
        print('Looking for scraped_L_feb_path', scraped_L_feb_path, os.path.exists(scraped_L_feb_path))
          
    if not os.path.exists(scraped_RL_feb_path) or not os.path.exists(scraped_L_feb_path):
        continue # skip if we don't have scraped free energies for dG2 and dG3
    
    # Note: all dG are in units kT 
    
    ####################### 
    # Compute dG1 -  the free energy of restraining the ligand in a solution
    #                at standard concentration (analytical)
    
    grofile = f'{prefix}/{dataset}_RL/RUN{run}/npt.gro'
    itpfile = f'{prefix}/{dataset}_RL/RUN{run}/LIG_res.itp'
    
    dG1, rest_function_used = harmonic_analytical.get_dG_harmonic_rest(grofile, itpfile)
    
    ####################### 
    # Compute dG2 -   free energy of decoupling the ligand in solution (EE)
    
    L_increments = np.load(f'{prefix}/results/{dataset}/scraped_data/{dataset}_L_RUN{run}_inc.npy')
    L_fe = np.load(f'{prefix}/results/{dataset}/scraped_data/{dataset}_L_RUN{run}_feb.npy')[:,-1]
    
    increment_threshold = 0.1
    
    for x,i in enumerate(L_increments):
        if i < increment_threshold:
            break
    #print(np.shape(L_fe), x, np.average(L_fe[x:]), L_fe[-1]) # < --- decide which to use
    dG2, dG2_sigma = np.average(L_fe[x:]), np.std(L_fe[x:])
    
    ####################### 
    # Compute dG3 -   free energy of *coupling* the ligand bound to the receptor
    #                 in the presence of restraints (EE)
    
    RL_increments = np.load(f'{prefix}/results/{dataset}/scraped_data/{dataset}_RL_RUN{run}_inc.npy')
    RL_fe = np.load(f'{prefix}/results/{dataset}/scraped_data/{dataset}_RL_RUN{run}_feb.npy')[:,-1]
                    
    for x,i in enumerate(RL_increments):
        if i < increment_threshold:
            last_frames = x*10 # un-stride for later
            break

    dG3, dG3_sigma = -np.average(RL_fe[x:]), np.std(RL_fe[x:])
    
    
    ####################### 
    # Compute dG4 -   is the free energy of removing the three position 
    #                 restraints on the ligand bound to the receptor (MBAR)
    
    ### Get the restraint indices from the itpfile
    fin = open(itpfile, 'r')
    lines = fin.readlines()
    fin.close()
    
    rest_lines = [line for line in lines if (line[0] != ';' and line[0] != '[')]
    rest_indices = []
    for line in rest_lines:
        fields = line.strip().split()
        rest_indices.append( int(fields[0]) )
    if verbose:
        print('rest_indices', rest_indices)

    
    # Make sure we have the starting grofile and xtc traj file
    xtc_path = f'{prefix}/{dataset}_RL/RUN{run}/traj_comp.xtc'
    if not os.path.exists(xtc_path):
        raise Exception(f"Can't find {xtc_path}!")
    npt_gro_path = f'{prefix}/{dataset}_RL/RUN{run}/npt.gro'
    if not os.path.exists(npt_gro_path):
        raise Exception(f"Can't find {npt_gro_path}!")
    
    # check to see if we have built a gro file for the xtc
    xtc_gro_path = f'{prefix}/{dataset}_RL/RUN{run}/xtc.gro'
    if not os.path.exists(xtc_gro_path):
        
        if verbose:
            print(f'Could not find {xtc_gro_path} , now making one...')
        # check to see if the current index file has the Protein_LIG atom group
        index_path = f'{prefix}/{dataset}_RL/RUN{run}/index.ndx'
        index_xtc_path = f'{prefix}/{dataset}_RL/RUN{run}/xtc.ndx'
        fin = open(index_path, 'r')
        index_text = fin.read()
        fin.close()
        
        # Find the indices of the [ Protein ] and [ LIG ] the atom groups
        directives = [line for line in index_text.split('\n') if (line.count('[ ') > 0)]
        protein_index = directives.index('[ Protein ]')
        lig_index = directives.index('[ LIG ]')
        
        has_Protein_LIG_group = (index_text.count('Protein_LIG') > 0)
        if not has_Protein_LIG_group:
            # make a new atom_group for Protein_LIG
            cmd = f'echo "{protein_index}|{lig_index}\\nq\\n" | gmx make_ndx -n {index_path} -o {index_xtc_path}'
            if verbose:
                print('>>', cmd)    
            os.system(cmd)
            protein_lig_index = len(directives)  # the new one we just added          
        else:
            protein_lig_index = directives.index('[ Protein_LIG ]')

        has_Protein_LIG_group = (index_text.count('Protein_LIG') > 0)
        if not has_Protein_LIG_group:
            # make a new atom_group for Protein_LIG
            cmd = f'echo "{protein_index}|{lig_index}\\nq\\n" | gmx make_ndx -n {index_path} -o {index_xtc_path}'
            if verbose:
                print('>>', cmd)    
            os.system(cmd)
            protein_lig_index = len(directives)  # the new one we just added          
        else:
            protein_lig_index = directives.index('[ Protein_LIG ]')  
        
        # save a xtc.gro with these atoms
        cmd = f'echo "{protein_lig_index}\\n" | gmx editconf -f {npt_gro_path} -n {index_xtc_path} -o {xtc_gro_path}'
        if verbose:
            print('>>', cmd)    
        os.system(cmd)
        
    # Read in the traj_comp.xtc trajectory data      
    traj = md.load(xtc_path, top=xtc_gro_path)
    if debug:
        print('traj.xyz.shape', traj.xyz.shape)
    
    # slice out only the restrained atoms
    Ind = np.array([i-1 for i in rest_indices])
    traj_positions = traj.xyz[:,Ind,:]
    traj_initial_position = traj_positions[0,:,:]
    if debug:
        print('traj_positions', traj_positions)
    
    # compile all possible periodic translations
    pbc_vecs = traj.unitcell_vectors[0]  # there are pbc vecs for each *frame*, we just take from the first frame
    if debug:
        print('pbc_vecs', pbc_vecs)
    all_pbc_translations = []
    for i in [-1., 0., 1.]:
        for j in [-1., 0., 1.]:
            for k in [-1., 0., 1.]:
                all_pbc_translations.append(i*pbc_vecs[0,:] + j*pbc_vecs[1,:] + k*pbc_vecs[2,:])
    all_pbc_translations = np.array(all_pbc_translations)
    if debug:
        print('all_pbc_translations', all_pbc_translations)
        print('all_pbc_translations.shape', all_pbc_translations.shape)
    
    # correct each frame
    traj_positions_corrected = np.zeros(traj_positions.shape)
    nframes, nparticles, ndims = traj_positions.shape[0], traj_positions.shape[1], traj_positions.shape[2]
    for i in range(nframes):
        for j in range(nparticles):
            all_images = all_pbc_translations + traj_positions[i,j,:]
            all_displacements_from_initial = all_images - traj_initial_position[j,:]
            all_sqdistances = np.sum(all_displacements_from_initial*all_displacements_from_initial, axis=1)
            # pick the closest image
            traj_positions_corrected[i,j,:] = all_images[np.argmin(all_sqdistances),:]
    
    # compute the distances
    traj_distances = np.zeros( (nframes, nparticles) )
    for j in range(nparticles):
        traj_displacements = traj_positions_corrected[:,j,:] - traj_positions_corrected[0,j,:]
        traj_distances[:,j] = np.sqrt(np.sum(traj_displacements*traj_displacements, axis=1))
    
    # plot distances over time
    plt.figure(figsize=(6,4))
    for j in range(nparticles):
        plt.plot(np.arange(nframes), traj_distances[:,j])
    plt.xlabel('frame')
    plt.ylabel('distance (nm)')
    plt.tight_layout()
    plt.savefig(f'{prefix}/results/{dataset}/plots/restraint_distance_trajectory_RUN{run}.png')
    

    plt.figure(figsize=(4,4))
    for j in range(nparticles):
        counts, bin_edges = np.histogram(traj_distances[:,j], bins=100, normed=True)
        bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2.0
        plt.plot(bin_centers, counts, label=f'particle {j}')
    plt.legend(loc='best')
    plt.title('distances from restrained positions (nm)')
    plt.xlabel('distance (nm)')
    plt.tight_layout()
    plt.savefig(f'{prefix}/results/{dataset}/plots/restraint_distance_histogram_RUN{run}.png')
    
    total_sqdisp_from_first_frame = np.sum(traj_distances*traj_distances, axis=1)
    if debug:
        print('total_sqdisp_from_first_frame', total_sqdisp_from_first_frame)
        print('np.max(total_sqdisp_from_first_frame)', np.max(total_sqdisp_from_first_frame))
    
    RT = 2.479 # RT in in kJ/mol at 298 K
    kvalue = 800.0 # kJ/mol/nm^2
    du_10 = -1.0*kvalue/2.0*total_sqdisp_from_first_frame/RT  # reduced du's of turning the restraint off.
    if debug:
        print('du_10', du_10)
        print('np.min(du_10)', np.min(du_10))
        print('np.mean( np.exp(-1.0*du_10)', np.mean( np.exp(-1.0*du_10)))
    
    plt.figure(figsize=(4,4))
    counts, bin_edges = np.histogram(du_10, bins=100)
    bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2.0
    plt.plot(bin_centers, counts)
    plt.xlabel('energy (kT)')
    plt.tight_layout()
    plt.savefig(f'{prefix}/results/{dataset}/plots/BAR_RUN{run}.png')
    
    ### DONT USE EXP averageing for free energy estimation  -- to o much bias
    dG4 = -1.0*np.log( np.mean( np.exp(-1.0*du_10) ) )
    if debug:
        print('dG4_EXP', dG4)
    du_10_subsampled = du_10[::20]  # subsample to 5%
    dG4_BAR, dG4_BAR_sigma = pymbar.BAR(du_10_subsampled, np.array([0]), DeltaF=0.0, compute_uncertainty=True, uncertainty_method='BAR')
    #               maximum_iterations=500, relative_tolerance=1e-12,
    #               verbose=False, method='false-position', iterated_solution=True, return_dict=False)
    if verbose:
        print('dG4_BAR', dG4_BAR)

    
    dG_binding = dG1 + dG2 + dG3 + dG4_BAR
    dG_binding_sigma = np.sqrt( dG2_sigma**2 + dG2_sigma**2 )
    Kd = np.exp(dG_binding)
    
    log_Kd_sigma = dG_binding_sigma/np.log(10)
    
    print('------------------------------------------------------------')
    print(f'{dataset}/RUN{run}')
    print('------------------------------------------------------------')
    print(f'dG1 = \t{dG1}')
    print(f'dG2 = \t{dG2} +/- {dG2_sigma}')
    print(f'dG3 = \t{dG3} +/- {dG3_sigma}')
    print(f'dG4 = \t{dG4_BAR} +/- {dG4_BAR_sigma}')
    print('-----------')
    print(f'dG_binding (units RT)   = {dG_binding}')
    print(f'dG_binding_uncertainty  = {dG_binding_sigma}')
    print(f'Kd                      = %3.2e M'%Kd)
    print(f'log_10 Kd               = %3.2f'%np.log10(Kd) )
    print(f'log_10 Kd uncertainty   = %3.2f'%log_Kd_sigma )
    print('------------------------------------------------------------')
    print('\n\n\n')

    
    data.append([f'{dataset}_{run}', dG1, dG2, dG3, dG4_BAR, dG_binding, np.log10(Kd),
                 dG2_sigma, dG3_sigma, dG4_BAR_sigma, dG_binding_sigma, log_Kd_sigma,
                 RL_increments[-1], L_increments[-1], np.shape(RL_increments)[0]/100, np.shape(L_increments)[0]/10])
df = pd.DataFrame(data, columns=['run', 'dG1', 'dG2', 'dG3', 'dG4', 'FEB', 'Kd',
                 'dG2_sigma', 'dG3_sigma', 'dG4_sigma', 'FEB_sigma', 'log_Kd_sigma',
                 'min_RL_increment', 'min_L_increment', 'RL_ns', 'L_ns'])
df.to_pickle(f'{prefix}/results/{dataset}/results.pkl')
