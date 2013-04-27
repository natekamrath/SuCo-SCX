"""
This module contains the survivor selection algorithms
"""
import random

def tournament(individuals, number, constants):
	"""
	Standard k tournament selection operator without replacement

	Parameters:

	-``individuals``: The population of individuals which can be chosen from.

	-``number``: The number of individuals to pick for survival.

	-``constants``: Config settings for the EA from the config files specified at run time.
	"""
	while len(individuals) > number:
		# Get the indexes of the tournament competitors
		tourn = random.sample(xrange(len(individuals)), min(len(individuals), constants["survivorTournament"]))
		# find the minimum, pairing individual and index
		toRemove = min((individuals[i], i) for i in tourn)
		# Swap the last guy up to the worst guy's position
		individuals[toRemove[1]], individuals[-1] = individuals[-1], None
		# shorten the list
		del individuals[-1]
	return individuals

def elitist(individuals, number, constants={}):
	"""
	Standard elitist selection operator

	Parameters:

	-``individuals``: The population of individuals which can be chosen from.

	-``number``: The number of individuals to pick for survival.

	-``constants``: Config settings for the EA from the config files specified at run time.  Defaults to an empty dictionary
	"""
	return sorted(individuals)[-number:]
