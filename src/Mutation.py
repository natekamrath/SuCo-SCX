"""
This module contains the mutation algorithms.
"""
import random
import Crossover
import Util
import copy
import math

def bitFlip(child, constants, parents=None):
	"""
	returns a child mutated by a bit flip mutation.

	Parameters:

	-``child``: The child individual to be mutated.

	-``constants``: Config settings for the EA from the config files specified at run time.

	-``parents``:  the parents ``child``.  Defaults to None.
	"""
	rate = constants["mutationRate"]
	for g in range(len(child.genes)):
		if random.random() < rate:
			child.genes[g] = Util.flip(child.genes[g])

def creep(child, constants, parents=None):
	"""
	returns a child mutated by a creep mutation.

	Parameters:

	-``child``: The child individual to be mutated.

	-``constants``: Config settings for the EA from the config files specified at run time.

	-``parents``:  the parents ``child``.  Defaults to None.
	"""
	step = constants["mutationStepSize"]
	rate = constants["mutationRate"]
	child.genes = [g + random.normalvariate(0, step) if random.random() < rate else g for g in child.genes]

def gaussian(child, constants, parents=None):
	"""
	returns a child mutated by a gaussian mutation.

	Parameters:

	-``child``: The child individual to be mutated.

	-``constants``: Config settings for the EA from the config files specified at run time.

	-``parents``:  the parents ``child``.  Defaults to None.
	"""
	step = constants["mutationStepSize"]
	rate = constants["mutationRate"]
	for g in range(len(child.genes)):
		if random.random() < rate: #comment this part out and mutate every gene performance offset rast much better
			child.genes[g] += random.gauss(0, step)

def scxOriginal(child, constants, parents=None):
	"""
	Special SCX operator mutation function.

	Parameters:

	-``child``: The child SCX individual to be mutated.

	-``constants``: Config settings for the EA from the config files specified at run time.

	-``parents``:  the parents ``child``.  Defaults to None.
	"""
	rate = constants["mutationRate"] * constants['dimensions'] / len(child.crossover.genes)
	for g in range(len(child.crossover.genes)):
		if random.random() < rate:
			child.crossover.genes[g] = random.choice(Util.childClasses(Crossover.Primitive))()

def gaussRates(child, constants, stepSizes):
	"""
	returns a child mutated by a gaussian mutation with custom step sizes

	Parameters:

	-``child``: The child individual to be mutated.

	-``constants``: Config settings for the EA from the config files specified at run time.

	-``stepSizes``:  a list of stepSizes to be used in the mutation of each individual gene.
	"""
	rate = constants["mutationRate"]
	for g in range(len(child.genes)):
		if random.random() < rate: #added
			child.genes[g] += random.gauss(0, stepSizes[g])

def tau(child, constants, parents=None):
	"""
	returns a child mutated by a tau mutation.

	Parameters:

	-``child``: The child individual to be mutated.

	-``constants``: Config settings for the EA from the config files specified at run time.

	-``parents``:  the parents ``child``.  Defaults to None.
	"""
	bias = 1 / math.sqrt(2.0 * constants["dimensions"]) * random.gauss(0, 1)
	tau = 1 / math.sqrt(2.0 * math.sqrt(constants["dimensions"]))
	child.genes = [gene * math.exp(bias + tau * random.gauss(0, 1)) for gene in child.genes]
