"""
This module contains fitness functions which can be used as benchmark problems to test EAs.
"""

import math
import random
import Population
import sys
import Experiments
import Util
import operator
import copy

class FitnessFunction(object):
	"""
	Base class for fitness functions
	"""
	def __init__(self, constants={}):
		"""
		Creates a new fitness function instance.
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time.  Defaults to an empty dictionary.
		"""
		self.constants = constants

	def eval(self, child):
		"""
		Evaluates a child or possible solution.
		
		Parameters:
		
		-``child``: The child to be evaluated by the fitness function.
		"""
		raise Exception("Fitness function did not override eval")

class Rastrigin(FitnessFunction):
	"""
	Class for the Rastrigin function.  http://en.wikipedia.org/wiki/Rastrigin_function
	"""
	def __init__(self, constants):
		"""
		Creates a new rastrigin instance.
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time.
		"""
		self.A = constants["scalar"]
	def eval(self, child):
		"""
		Evaluates a child or possible solution.
		
		Parameters:
		
		-``child``: The child to be evaluated by the fitness function.
		"""
		child.fitness = round(-sum(map(self.function, child.genes)), 6)
	def function(self, x):
		"""
		The fitness function
		
		Parameters:
		
		-``x``: The solution to be evaluated by the function.
		"""
		return self.A + x * x - self.A * math.cos(2 * math.pi * x)

class ShiftedRastrigin(Rastrigin):
	def __init__(self, constants):
		"""
		Creates a new shifted rastrigin instance.
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time.
		"""
		self.A = constants["scalar"]
		rngstate = random.getstate()
		number = constants['problemSeed'] + constants["runNumber"]
		random.seed(number)
		self.offset = [random.uniform(constants["min"], constants["max"]) for _ in range(constants["dimensions"])]
		random.seed(rngstate)

	def eval(self, child):
		"""
		Evaluates a child or possible solution.
		
		Parameters:
		
		-``child``: The child to be evaluated by the fitness function.
		"""
		z = [x + y for x, y in zip(self.offset, child.genes)]
		child.fitness = round(-sum(map(self.function, z)), 6)

class OffsetRastrigin(Rastrigin):
	def __init__(self, constants):
		"""
		Creates a new offset rastrigin instance.
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time.
		"""
		self.A = constants["scalar"]
		rngstate = random.getstate()
		number = constants['problemSeed'] + constants["runNumber"]
		random.seed(number)
		self.offset = [random.randint(0, 10) * 0.5 - 2.5 for _ in range(constants["dimensions"])]
		random.seed(rngstate)

	def eval(self, child):
		"""
		Evaluates a child or possible solution.
		
		Parameters:
		
		-``child``: The child to be evaluated by the fitness function.
		"""
		z = [x + y for x, y in zip(self.offset, child.genes)]
		child.fitness = round(-sum(map(self.function, z)), 6)

class Rosenbrock(FitnessFunction):
	"""
	Class for the Rosenbrock function.  http://en.wikipedia.org/wiki/Rosenbrock_function
	"""
	def __init__(self, constants):
		"""
		Creates a new rosenbrock instance.
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time.
		"""
		self.A = constants["scalar"]

	def function(self, genes, i):
		"""
		Defines the rosenbrock function.
		
		Parameters:
		
		-``genes``:  List of genes being evaluated.
		
		-``i``: The index of a specific gene within the solution genes.
		"""
		return (1 - genes[i]) ** 2 + self.A * (genes[i + 1] - genes[i] ** 2) ** 2

	def eval(self, child):
		"""
		Evaluates a child or possible solution.
		
		Parameters:
		
		-``child``: The child to be evaluated by the fitness function.
		"""
		child.fitness = 0
		for i in range(len(child.genes) - 1):
			child.fitness -= self.function(child.genes, i)
		child.fitness = round(child.fitness, 6)

class ShiftedRosenbrock(Rosenbrock):
	def __init__(self, constants):
		"""
		Creates a new shifted rosenbrock instance.
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time.
		"""
		self.A = constants["scalar"]
		rngstate = random.getstate()
		number = constants['problemSeed'] + constants["runNumber"]
		random.seed(number)
		self.offset = [random.uniform(1 - constants["max"], 1 - constants["min"]) for _ in range(constants["dimensions"])]
		random.seed(rngstate)

	def eval(self, child):
		"""
		Evaluates a child or possible solution.
		
		Parameters:
		
		-``child``: The child to be evaluated by the fitness function.
		"""
		child.fitness = 0
		z = [x + y for x, y in zip(self.offset, child.genes)]
		for i in range(len(child.genes) - 1):
			child.fitness -= self.function(z, i)
		child.fitness = round(child.fitness, 6)
