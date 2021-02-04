#!/usr/bin/env python3

import argparse
from heapq import nsmallest
from math import log
import numpy as np
from sklearn.linear_model import LinearRegression

argument_parser = argparse.ArgumentParser(description = 'Molecular Dynamics Residence Time')
argument_parser.add_argument('data_file', type = str, help = 'Data file')
argument_parser.add_argument('dump_file', type = str, help = 'Dump file')
argument_parser.add_argument('adsorbent_atom_id_start', type = int, help = 'Adsorbent atom id start (inclusive)')
argument_parser.add_argument('adsorbent_atom_id_end', type = int, help = 'Adsorbent atom id end (inclusive)')
argument_parser.add_argument('adsorbate_atom_id_start', type = int, help = 'Adsorbate atom id start (inclusive)')
argument_parser.add_argument('adsorbate_atom_id_end', type = int, help = 'Adsorbate atom id end (inclusive)')
argument_parser.add_argument('--adsorption_threshold', type = int, default = -1, help = 'Threshold for the number of molecules considered adsorbed on the adsorbent at any instant of time (defaults to half the number of adsorbate molecules)')

args = argument_parser.parse_args()
data_file = args.data_file
dump_file = args.dump_file
adsorbent_atom_id_start = args.adsorbent_atom_id_start
adsorbent_atom_id_end = args.adsorbent_atom_id_end
adsorbate_atom_id_start = args.adsorbate_atom_id_start
adsorbate_atom_id_end = args.adsorbate_atom_id_end
adsorption_threshold = args.adsorption_threshold

def is_adsorbent_atom(atom_id):
	return atom_id >= adsorbent_atom_id_start and atom_id <= adsorbent_atom_id_end

def is_adsorbate_atom(atom_id):
	return atom_id >= adsorbate_atom_id_start and atom_id <= adsorbate_atom_id_end

atom_id_to_mol_id = {}
mols_continuously_remained_adsorbed = set()
with open(data_file, newline = '') as datafile:
	for line in datafile:
		if line == 'Atoms\n':
			break
	datafile.readline()
	for line in datafile:
		if line == '\n':
			break
		atom_id, mol_id, _, _, _, _, _ = line.split()
		atom_id = int(atom_id)
		mol_id = int(mol_id)
		atom_id_to_mol_id[atom_id] = mol_id
		if is_adsorbate_atom(atom_id):
			mols_continuously_remained_adsorbed.add(mol_id)

if adsorption_threshold == -1:
	adsorption_threshold = len(mols_continuously_remained_adsorbed) // 2

initially_adsorbed_mols = 0		# Will be initialized after the first timestep data gets processed
log_auto_correlation = []
timesteps = []
with open(dump_file, newline = '') as dumpfile:
	while dumpfile.readline():
		timestep = int(dumpfile.readline().strip())
		dumpfile.readline()
		num_atoms = int(dumpfile.readline().strip())

		for _ in range(5):
			dumpfile.readline()
		
		adsorbent_mols_avg_coords = [{}, {}, {}]
		adsorbate_mols_avg_coords = [{}, {}, {}]
		for i in range(num_atoms):
			coords = [0] * 3
			atom_id, _, coords[0], coords[1], coords[2], _, _, _ = dumpfile.readline().split()
			atom_id = int(atom_id)
			for j in range(3):
				coords[j] = float(coords[j])

			if is_adsorbent_atom(atom_id):
				mol_id = atom_id_to_mol_id[atom_id]
				for j in range(3):
					if mol_id in adsorbent_mols_avg_coords[j]:
						adsorbent_mols_avg_coords[j][mol_id][0] += coords[j]
						adsorbent_mols_avg_coords[j][mol_id][1] += 1
					else:
						adsorbent_mols_avg_coords[j][mol_id] = [coords[j], 1]
			
			if is_adsorbate_atom(atom_id):
				mol_id = atom_id_to_mol_id[atom_id]
				for j in range(3):
					if mol_id in adsorbate_mols_avg_coords[j]:
						adsorbate_mols_avg_coords[j][mol_id][0] += coords[j]
						adsorbate_mols_avg_coords[j][mol_id][1] += 1
					else:
						adsorbate_mols_avg_coords[j][mol_id] = [coords[j], 1]
				
		adsorbent_avg_coords = [0] * 3
		for i in range(3):
			for _, mol_avg in adsorbent_mols_avg_coords[i].items():
				adsorbent_avg_coords[i] += mol_avg[0] / mol_avg[1]
		for i in range(3):
			adsorbent_avg_coords[i] /= len(adsorbent_mols_avg_coords[i])

		distances_and_ids = []
		for mol_id in adsorbate_mols_avg_coords[0]:
			dist = 0
			for i in range(3):
				mol_avg = adsorbate_mols_avg_coords[i][mol_id]
				dist += (mol_avg[0] / mol_avg[1] - adsorbent_avg_coords[i]) ** 2
			distances_and_ids.append((dist, mol_id))
		
		closest_distances_and_ids = nsmallest(adsorption_threshold, distances_and_ids)

		new_set = set()
		for _, mol_id in closest_distances_and_ids:
			if mol_id in mols_continuously_remained_adsorbed:
				new_set.add(mol_id)
		mols_continuously_remained_adsorbed = new_set

		if timestep == 0:
			initially_adsorbed_mols = len(mols_continuously_remained_adsorbed)
		
		log_auto_correlation.append(log(len(mols_continuously_remained_adsorbed) / initially_adsorbed_mols))
		timesteps.append(timestep)

reg = LinearRegression().fit(np.asarray(timesteps).reshape(-1, 1), np.asarray(log_auto_correlation).reshape(-1, 1))

print(f"Calculated residence time (in timesteps):", -1 / reg.coef_[0, 0])
