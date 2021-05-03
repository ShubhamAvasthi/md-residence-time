#!/usr/bin/env python3

import argparse
import numpy as np
from math import inf
from sklearn.linear_model import LinearRegression
from tqdm import trange

# Argparse initializations

argument_parser = argparse.ArgumentParser(description = 'Molecular Dynamics Residence Time')
argument_parser.add_argument('data_file', type = str, help = 'Data file')
argument_parser.add_argument('dump_file', type = str, help = 'Dump file')
argument_parser.add_argument('adsorbent_atom_id_start', type = int, help = 'Adsorbent atom id start (inclusive)')
argument_parser.add_argument('adsorbent_atom_id_end', type = int, help = 'Adsorbent atom id end (inclusive)')
argument_parser.add_argument('adsorbate_atom_id_start', type = int, help = 'Adsorbate atom id start (inclusive)')
argument_parser.add_argument('adsorbate_atom_id_end', type = int, help = 'Adsorbate atom id end (inclusive)')

args = argument_parser.parse_args()
data_file = args.data_file
dump_file = args.dump_file
adsorbent_atom_id_start = args.adsorbent_atom_id_start
adsorbent_atom_id_end = args.adsorbent_atom_id_end
adsorbate_atom_id_start = args.adsorbate_atom_id_start
adsorbate_atom_id_end = args.adsorbate_atom_id_end

# Some helper functions

def is_adsorbent_atom(atom_id):
	return atom_id >= adsorbent_atom_id_start and atom_id <= adsorbent_atom_id_end

def is_adsorbate_atom(atom_id):
	return atom_id >= adsorbate_atom_id_start and atom_id <= adsorbate_atom_id_end

def squared_distance(coords1, coords2):
	dist = 0
	for i in range(3):
		dist += (coords1[i] - coords2[i]) ** 2
	return dist

# Helper class to find average coordinates of a set of molecules

class AverageCoords:
	
	def __init__(self, coords):
		self.coords = coords
		self.num = 1
	
	def add_contribution(self, coords):
		for i in range(len(self.coords)):
			self.coords[i] = (self.coords[i] * self.num + coords[i]) / (self.num + 1)
		self.num += 1

# Initializations using the data file

adsorbate_mols = []
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
		if is_adsorbate_atom(atom_id):
			adsorbate_mols.append(mol_id)

# Helper class to store dump file data

class AtomData:

	def __init__(self, atom_id, coords):
		self.atom_id = atom_id
		self.coords = coords

# Initializations using the dump file

timesteps = []
dump_file_data = []

with open(dump_file, newline = '') as dumpfile:
	while dumpfile.readline():
		timestep = int(dumpfile.readline().strip())
		dumpfile.readline()
		num_atoms = int(dumpfile.readline().strip())

		for _ in range(5):
			dumpfile.readline()

		timesteps.append(timestep)
		dump_file_data.append([])
		
		adsorbent_atoms_coords = []
		adsorbate_mols_avg_coords = {}

		for i in range(num_atoms):
			coords = [0] * 3
			atom_id, _, coords[0], coords[1], coords[2], _, _, _ = dumpfile.readline().split()
			atom_id = int(atom_id)
			for j in range(3):
				coords[j] = float(coords[j])
			dump_file_data[-1].append(AtomData(atom_id, coords))

# Residence time calculation

num = [0] * len(timesteps)
auto_correlation_avgs = [0] * len(timesteps)

for t0 in trange(len(timesteps), desc = 'Overall progress'):
	mols_continuously_remained_adsorbed = set(adsorbate_mols)
	auto_correlation = []
	
	for ts_index in trange(t0, len(timesteps), leave = False, desc = 'Step progress   '):
		timestep = timesteps[ts_index]
		adsorbent_atoms_coords = []
		adsorbate_mols_avg_coords = {}

		for i in range(len(dump_file_data[ts_index])):
			atom_id = dump_file_data[ts_index][i].atom_id
			coords = dump_file_data[ts_index][i].coords

			if is_adsorbent_atom(atom_id):
				adsorbent_atoms_coords.append(coords)
			
			if is_adsorbate_atom(atom_id):
				mol_id = atom_id_to_mol_id[atom_id]
				if mol_id in adsorbate_mols_avg_coords:
					adsorbate_mols_avg_coords[mol_id].add_contribution(coords)
				else:
					adsorbate_mols_avg_coords[mol_id] = AverageCoords(coords)

		new_adsorbed_set = set()
		for adsorbent_coords in adsorbent_atoms_coords:
			min_dist = inf
			closest_adsorbate_mol_id = -1
			for adsorbate_mol_id, adsorbate_avg_coords in adsorbate_mols_avg_coords.items():
				dist = squared_distance(adsorbent_coords, adsorbate_avg_coords.coords)
				if dist < min_dist:
					min_dist = dist
					closest_adsorbate_mol_id = adsorbate_mol_id
			if closest_adsorbate_mol_id in mols_continuously_remained_adsorbed:
				new_adsorbed_set.add(closest_adsorbate_mol_id)
		mols_continuously_remained_adsorbed = new_adsorbed_set

		if ts_index == t0:
			initially_adsorbed_mols = len(mols_continuously_remained_adsorbed)
		
		auto_correlation.append(len(mols_continuously_remained_adsorbed) / initially_adsorbed_mols)

	while auto_correlation and auto_correlation[-1] == 0:
		auto_correlation.pop()

	for i in range(len(auto_correlation)):
		auto_correlation_avgs[t0 + i] *= num[t0 + i]
		auto_correlation_avgs[t0 + i] += auto_correlation[i]
		num[t0 + i] += 1
		auto_correlation_avgs[t0 + i] /= num[t0 + i]

auto_correlation_avgs = np.array(np.log(auto_correlation_avgs)).reshape(-1, 1)
timesteps = np.array(timesteps).reshape(-1, 1)

print("Calculated residence time (in timesteps):", -1 / LinearRegression().fit(timesteps, auto_correlation_avgs).coef_[0, 0])
