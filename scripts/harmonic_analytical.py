import os, sys
import numpy as np



def dG_harmonic_rest_single(k=800.0, L=1.184048):
    """Returns the free energy of turning on a single harmonic restraint in units RT

    PARAMETERS

    k      - the force constant, in kJ/mol/nm^2  (Default: 800.0 kJ/mol/nm^2)
    L      - the length of the cubic box (Default: 1.184048 nm ) 

                V0 = 1660.0                   # in Angstrom^3 is the standard volume
                L0 = ((1660.0)**(1/3)) * 0.1  # converted to nm
                L0 = 1.1840481475428983 nm

    OUTPUT

    dG_in_kT    - the reduced free energy of turning *on* the restraint (i.e. in units kT)
    """

    RT      = 2.479           # in kJ/mol at 298 K

    dG_in_kT = -1.0*( 3.0/2.0*np.log(2.0*np.pi*RT/k) 
                        - 3.0*np.log(L) )

    return dG_in_kT


def dG_harmonic_rest_double(x1, x2, k=800.0, L=1.184048):
    """Returns the free energy of turning on a double harmonic restraint in units RT, 
    for a rigid molecule restrained at anchors x1 and x2.

    INPUTS

    x1, x2 - the coordinates of the two restraint positions as 3-element np.array(),   in units nm

    PARAMETERS

    k      - the force constant, in kJ/mol/nm^2  (Default: 800.0 kJ/mol/nm^2)
    L      - the length of the cubic box (Default: 1.184048 nm ) 

                V0 = 1660.0                   # in Angstrom^3 is the standard volume
                L0 = ((1660.0)**(1/3)) * 0.1  # converted to nm
                L0 = 1.1840481475428983 nm


    """

    RT      = 2.479           # in kJ/mol at 298 K


    # Calculate the the distance d between the two positions (in nm)
    d = np.sqrt( np.dot(x2-x1,x2-x1) )

    dG_in_kT =   -1.0*(3.0/2.0*np.log(2.0*np.pi*ee.RT/(3.0*k)))   # three translational d.o.f.
    dG_in_kT +=  -1.0*(2.0/2.0*np.log(2.0*np.pi*ee.RT/(2.0*k)))   # two rotational
    dG_in_kT +=  np.log((ee.L)**3 * 4.0 * np.pi * (ee.d/2.0)**2 )  # demoninator

    return dG_in_kT


def dG_harmonic_rest_triple(x1, x2, x3, k=800.0, L=1.184048):
    """Returns the free energy of turning on a triple harmonic restraint in units RT, 
    for a rigid nonlinear molecule restrained at anchors x1, x2, and x3

    INPUTS

    x1, x2, x3 - the coordinates of the three restraint positions as 3-element np.array(), each in units nm

    PARAMETERS

    k      - the force constant, in kJ/mol/nm^2  (Default: 800.0 kJ/mol/nm^2)
    L      - the length of the cubic box (Default: 1.184048 nm ) 

                V0 = 1660.0                   # in Angstrom^3 is the standard volume
                L0 = ((1660.0)**(1/3)) * 0.1  # converted to nm
                L0 = 1.1840481475428983 nm


    """

    RT      = 2.479           # in kJ/mol at 298 K


    # Calculate distance between particles 1 and 2
    d = np.sqrt( np.dot(x2-x1,x2-x1) )
        
    # Calculate the altitude (height), c, of triangle where p1-p2 is the base
    ### 1. First calculate the area of the triangle as A = (1/2)||v \cross w||
    v, w = x2-x1, x3-x1
    area = 0.5*np.linalg.norm( np.cross(v,w) )
    ### 2. Then since A = 1/2 * base * height, the height is c = 2*area/base
    c = 2.0 * area / np.linalg.norm(v)
        
    # Calculate e, the projection of x3-x1 = w in the x2-x1 = v direction
    unit_vec_along_x12 = v / d
    e = np.abs(np.dot(w, unit_vec_along_x12))
        
    if verbose:
        print('Distance betweeen x1 and x2, \td =', d, 'nm')
        print('Height of x3 from triangular base x1-x2, \tc =', c, 'nm')
        print('Projection of x3-x1 in the x2-x1 direction, \te =', e, 'nm')

    kc_coeff = 1.0 + (e/d)**2.0
    if verbose:
        print('kc_coeff', kc_coeff)
    kp1_coeff = 1.0 + (c**2 + e**2)/(d**2)
    if verbose:
        print('kp1_coeff', kp1_coeff)

    '''
        theory_dG_in_kT = -1.0*( 3.0/2.0*np.log(2.0*np.pi*ee.RT/(3.0*ee.k_values[1:]))        \ # translational
                                + 1.0/2.0*np.log(2.0*np.pi*ee.RT/(2.0*ee.k_values[1:]))       \ # rot about d
                                + 1.0/2.0*np.log(2.0*np.pi*ee.RT/(kc_coeff*ee.k_values[1:]))  \ # rot about c
                                + 1.0/2.0*np.log(2.0*np.pi*ee.RT/(kp1_coeff*ee.k_values[1:])) \ # rot out of page
                                - np.log( ee.L**3 * 8.0 * (np.pi**2) * (ee.d)**2 * ee.c  ) )
    '''

    theory_dG_in_kT = -1.0*( 3.0/2.0*np.log(2.0*np.pi*RT/(3.0*k))            \
                                + 1.0/2.0*np.log(2.0*np.pi*RT/k)             \
                                + 1.0/2.0*np.log(2.0*np.pi*RT/(kc_coeff*k))  \
                                + 1.0/2.0*np.log(2.0*np.pi*RT/(kp1_coeff*k)) \
                                - np.log( L**3 * 8.0 * (np.pi**2) * d**2 * c  ) )
    return theory_dG_in_kT


########## Main program -- use as script ##########

if __name__ == '__main__':

    usage = '''Usage:     python harmonic_analytical.py  grofile  restraint_itpfile

    DESCRIPTION

    Given a *.gro file (grofile),
    and a LIG_res.itp file with list of atom_indices (gmx-numbered, starting at 1),
    this script will return the standard reduced free energy (units kT) of
    turning on 1, 2 or 3 harmonic restraints (depending on the number found in the *.itp), 
    assuming the restrained ligand is a rigid body.

    NOTES

    The LIG_res.itp should look like this:

       ;LIG_res.itp
       [ position_restraints ]
       ;i funct       fcx        fcy        fcz
       14    1        800        800        800
       16    1        800        800        800
       22    1        800        800        800%              

    Left column is atom index, 3rd (and 4th, 5th) column is the harmonic force constant, k (kJ/mol/nm^2)


    EXAMPLES

    Try:
        $ python python harmonic_analytical.py ../x11294_RL/RUN0/xtc.gro  ../x11294_RL/RUN0/LIG_res.itp

    '''

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)


    grofile = sys.argv[1]
    itpfile = sys.argv[2]

    verbose = True

    # Parse the itpfile to get the restraint indices
    print('Reading', itpfile, '...')
    fin = open(itpfile, 'r')
    lines = fin.readlines()
    fin.close()

    rest_lines = [line for line in lines if (line[0] != ';' and line[0] != '[')]
    rest_indices = []
    for line in rest_lines:
        fields = line.strip().split()
        rest_indices.append( int(fields[0]) )
        k = float(fields[2])

    if len(rest_indices) < 1:
        raise Exception("Could not find any restraint indices in %s !"%itpfile)
    if len(rest_indices) > 3:
        raise Exception("There are more than three restraint indices in %s !"%itpfile)
    print('...Done.')

    print('Restraint indices:', rest_indices)


    # Parse the grofile to get the positions of the restraint atoms
    print('Reading', grofile, '...')
    fin = open(grofile, 'r')
    lines = fin.readlines()
    fin.close()

    '''Example grofile:

Generic title
4686
  305LIG      F    1   1.174   7.114   2.260
  305LIG      C    2   1.230   7.236   2.276
  305LIG      C    3   1.331   7.280   2.188
  305LIG      C    4   1.186   7.321   2.380
  305LIG      F    5   1.373   7.201   2.086
  305LIG      C    6   1.387   0.078   2.205
  305LIG      C    7   1.243   0.120   2.391
....
  304THR      C 4684   0.566   6.394   6.320
  304THR      O 4685   0.642   6.345   6.406
  304THR    OXT 4686   0.505   6.328   6.233
   7.32884   7.32884   7.32884
    '''
 
   
    # Assuming this is a standard grofile, pop title and natoms
    title = lines.pop(0)
    natoms = int(lines.pop(0))
    boxsize = lines.pop()

    positions = [ None ] * len(rest_indices)
    found_grolines = [ None ] * len(rest_indices)
    for line in lines:
        fields = line.strip().split()
        atom_index = int(fields[2])
        if rest_indices.count(atom_index) > 0:
            found_grolines[ rest_indices.index(atom_index) ] =  line
            x, y, z = float(fields[3]), float(fields[4]), float(fields[5])
            positions[ rest_indices.index(atom_index) ] = np.array([x,y,z])
    print('Found grolines', found_grolines) 

    if len(positions) == 1:
        dG_rest_in_kT = dG_harmonic_rest_single(k, L=1.184048)
        print('dG_rest_in_kT (single):', dG_rest_in_kT)
    elif len(positions) == 2:
        dG_rest_in_kT = dG_harmonic_rest_double(positions[0], positions[1], k, L=1.184048)
        print('dG_rest_in_kT (double):', dG_rest_in_kT)
    elif len(positions) == 3:
        dG_rest_in_kT = dG_harmonic_rest_triple(positions[0], positions[1], positions[2], k, L=1.184048)
        print('dG_rest_in_kT (triple):', dG_rest_in_kT)
    else:
        raise Exception('len(positions) = %d ???'%len(positions))

