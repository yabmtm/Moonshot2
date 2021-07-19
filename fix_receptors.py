#!/usr/bin/env python

from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile
import re, tqdm, glob

if 1:
    for lig in tqdm.tqdm(['P0009_RL']): #,'x11294_RL','x12692_RL']):
        receptor_file = f'{lig}/receptor.pdb'
        fixed_receptor_file = re.sub('receptor','fixed_receptor',receptor_file)
        fixer = PDBFixer(receptor_file) #filename='receptor.pdb')
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=True)
        missingatoms = fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        numChains = len(list(fixer.topology.chains()))
#        fixer.removeChains(range(1, numChains))        
        PDBFile.writeFile(fixer.topology, fixer.positions, open(fixed_receptor_file, 'w'))
