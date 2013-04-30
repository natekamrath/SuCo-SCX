"""
This module contains the parent selection algorithms
"""
import random

def tournament(individuals, number, constants):
	"""
	Standard k tournament selection operator with replacement

	Parameters:

	-``individuals``: The population of individuals which can be chosen from.

	-``number``: The number of individuals to pick for survival.

	-``constants``: Config settings for the EA from the config files specified at run time.
	"""
	return [max(random.sample(individuals, min(len(individuals), constants["parentTournament"])))
			for _ in range(number)]

def elitist(individuals, number, constants={}):
	"""
	Standard elitist selection operator

	Parameters:

	-``individuals``: The population of individuals which can be chosen from.

	-``number``: The number of individuals to pick for survival.

	-``constants``: Config settings for the EA from the config files specified at run time.  Defaults to an empty dictionary
	"""
	return sorted(individuals, reverse=True)[:number]
