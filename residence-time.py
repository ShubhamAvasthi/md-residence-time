#!/usr/bin/env python

# Questions:
# What are ix, iy and iz?

import argparse
from heapq import nsmallest
from math import log
import numpy as np
from sklearn.linear_model import LinearRegression

argument_parser = argparse.ArgumentParser(description = 'Molecular Dynamics Residence Time')
argument_parser.add_argument('data_file', type = str, help = 'Data file')
argument_parser.add_argument('dump_file', type = str, help = 'Dump file')
argument_parser.add_argument('adsorbent_mol_id_start', type = int, help = 'Adsorbent molecule id start (inclusive)')
argument_parser.add_argument('adsorbent_mol_id_end', type = int, help = 'Adsorbent molecule id end (inclusive)')
argument_parser.add_argument('adsorbate_mol_id_start', type = int, help = 'Adsorbate molecule id start (inclusive)')
argument_parser.add_argument('adsorbate_mol_id_end', type = int, help = 'Adsorbate molecule id end (inclusive)')
argument_parser.add_argument('--adsorption_threshold', type = int, default = -1, help = 'Threshold for the number of molecules considered adsorbed on the adsorbent at any instant of time (defaults to half the number of adsorbate molecules)')

args = argument_parser.parse_args()
data_file = args.data_file
dump_file = args.dump_file
adsorbent_mol_id_start = args.adsorbent_mol_id_start
adsorbent_mol_id_end = args.adsorbent_mol_id_end
adsorbate_mol_id_start = args.adsorbate_mol_id_start
adsorbate_mol_id_end = args.adsorbate_mol_id_end
adsorption_threshold = args.adsorption_threshold

atom_id_to_mol_id = {}
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

initially_adsorbed_mols = set()
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
		adsorbent_mols_atom_cnt = {}
		adsorbate_mols_avg_coords = [{}, {}, {}]
		adsorbate_mols_atom_cnt = {}
		for i in range(num_atoms):
			coords = [0] * 3
			atom_id, _, coords[0], coords[1], coords[2], _, _, _ = dumpfile.readline().split()
			atom_id = int(atom_id)
			for j in range(3):
				coords[j] = float(coords[j])
			
			mol_id = atom_id_to_mol_id[atom_id]

			if mol_id >= adsorbent_mol_id_start and mol_id <= adsorbent_mol_id_end:
				for j in range(3):
					if mol_id in adsorbent_mols_avg_coords[j]:
						adsorbent_mols_avg_coords[j][mol_id] += coords[j]
					else:
						adsorbent_mols_avg_coords[j][mol_id] = 0
			
			if mol_id >= adsorbate_mol_id_start and mol_id <= adsorbate_mol_id_end:
				for j in range(3):
					if mol_id in adsorbate_mols_avg_coords[j]:
						adsorbate_mols_avg_coords[j][mol_id] += coords[j]
					else:
						adsorbate_mols_avg_coords[j][mol_id] = 0
				if mol_id in adsorbate_mols_atom_cnt:
					adsorbate_mols_atom_cnt[mol_id] += 1
				else:
					adsorbate_mols_atom_cnt[mol_id] = 0
		
		if timestep == 0:
			if adsorption_threshold == -1:
				adsorption_threshold = len(adsorbate_mols_atom_cnt) // 2
		
		for mol_id, cnt in adsorbent_mols_atom_cnt:
			for i in range(3):
				adsorbent_mols_avg_coords[i][mol_id] /= cnt

		for mol_id, cnt in adsorbate_mols_atom_cnt.items():
			for i in range(3):
				adsorbate_mols_avg_coords[i][mol_id] /= cnt
		
		adsorbent_avg_coords = [0] * 3
		for mol_id, cnt in adsorbent_mols_atom_cnt.items():
			for i in range(3):
				adsorbent_avg_coords[i] += adsorbent_mols_avg_coords[i][mol_id]
		for mol_id, cnt in adsorbent_mols_atom_cnt:
			for i in range(3):
				adsorbent_avg_coords[i] /= len(adsorbent_mols_avg_coords[i])

		distances_and_ids = []
		for mol_id, cnt in adsorbate_mols_atom_cnt.items():
			dist = 0
			for i in range(3):
				dist += (adsorbate_mols_avg_coords[i][mol_id] - adsorbent_avg_coords[i]) ** 2
			distances_and_ids.append((dist, mol_id))
		
		closest_distances_and_ids = nsmallest(adsorption_threshold, distances_and_ids)

		if timestep == 0:
			for _, mol_id in closest_distances_and_ids:
				initially_adsorbed_mols.add(mol_id)
		
		num_still_adsorbed = 0
		for _, mol_id in closest_distances_and_ids:
			if mol_id in initially_adsorbed_mols:
				num_still_adsorbed += 1
		
		log_auto_correlation.append(log(num_still_adsorbed / len(initially_adsorbed_mols)))
		timesteps.append(timestep)

reg = LinearRegression().fit(np.asarray(timesteps).reshape(-1, 1), np.asarray(log_auto_correlation).reshape(-1, 1))

print(f"Calculated residence time (in timesteps):", -1 / reg.coef_[0, 0])
