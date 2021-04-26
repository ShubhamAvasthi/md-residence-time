#!/usr/bin/env python3

import argparse
import numpy as np
from math import inf, sqrt
from sklearn.linear_model import LinearRegression

from csv import DictWriter

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

def is_adsorbent_atom(atom_id):
	return atom_id >= adsorbent_atom_id_start and atom_id <= adsorbent_atom_id_end

def is_adsorbate_atom(atom_id):
	return atom_id >= adsorbate_atom_id_start and atom_id <= adsorbate_atom_id_end

def distance(coords1, coords2):
	dist = 0
	for i in range(3):
		dist += (coords1[i] - coords2[i]) ** 2
	return sqrt(dist)

class AverageCoords:
	
	def __init__(self, coords):
		self.coords = coords
		self.num = 1
	
	def add_contribution(self, coords):
		for i in range(len(self.coords)):
			self.coords[i] = (self.coords[i] * self.num + coords[i]) / (self.num + 1)
		self.num += 1


with open('out.csv', 'w', newline='') as outfile:
	d = DictWriter(outfile, fieldnames = [100 * i for i in range(1001)])

	for t0 in range(0, 100001, 100):
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

		initially_adsorbed_mols = -1		# Will be initialized after the first timestep data gets processed
		auto_correlation = []
		timesteps = []
		with open(dump_file, newline = '') as dumpfile:
			while dumpfile.readline():
				timestep = int(dumpfile.readline().strip())
				dumpfile.readline()
				num_atoms = int(dumpfile.readline().strip())

				for _ in range(5):
					dumpfile.readline()
				
				adsorbent_atoms_coords = []
				adsorbate_mols_avg_coords = {}
				for i in range(num_atoms):
					coords = [0] * 3
					atom_id, _, coords[0], coords[1], coords[2], _, _, _ = dumpfile.readline().split()
					atom_id = int(atom_id)
					for j in range(3):
						coords[j] = float(coords[j])

					if is_adsorbent_atom(atom_id):
						adsorbent_atoms_coords.append(coords)
					
					if is_adsorbate_atom(atom_id):
						mol_id = atom_id_to_mol_id[atom_id]
						if mol_id in adsorbate_mols_avg_coords:
							adsorbate_mols_avg_coords[mol_id].add_contribution(coords)
						else:
							adsorbate_mols_avg_coords[mol_id] = AverageCoords(coords)
				if timestep < t0:
					continue

				new_adsorbed_set = set()
				for adsorbent_coords in adsorbent_atoms_coords:
					min_dist = inf
					closest_adsorbate_mol_id = -1
					for adsorbate_mol_id, adsorbate_avg_coords in adsorbate_mols_avg_coords.items():
						dist = distance(adsorbent_coords, adsorbate_avg_coords.coords)
						if dist < min_dist:
							min_dist = dist
							closest_adsorbate_mol_id = adsorbate_mol_id
					if closest_adsorbate_mol_id in mols_continuously_remained_adsorbed:
						new_adsorbed_set.add(closest_adsorbate_mol_id)
				mols_continuously_remained_adsorbed = new_adsorbed_set

				if initially_adsorbed_mols == -1:
					initially_adsorbed_mols = len(mols_continuously_remained_adsorbed)
				
				auto_correlation.append(len(mols_continuously_remained_adsorbed) / initially_adsorbed_mols)
				timesteps.append(timestep)

		d.writerow({t0 + 100 * i: auto_correlation[i] for i in range(len(auto_correlation))})

		while auto_correlation and auto_correlation[-1] == 0:
			auto_correlation.pop()
			timesteps.pop()
		d.writerow({t0 + 100 * i: -1 / LinearRegression().fit(np.asarray(timesteps[:i+1]).reshape(-1, 1), np.log(auto_correlation[:i+1]).reshape(-1, 1)).coef_[0, 0] for i in range(len(auto_correlation))})
