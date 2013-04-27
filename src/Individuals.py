"""
This module contains classes for representing the individuals needed for the experiments
"""
import sys
import math

class Individual(object):
	"""
	Class for a solution individual with the potential to have an SCX individual.
	"""
	def __init__(self, genes=[], id=None, crossover=None, fitness= -sys.maxint - 1):
		"""
		Creates a new individual instance
		
		Parameters:
		
		-``genes``: The genes which the individual should be initialized with.  Defaults to an empty list.
		
		-``id``: The initial id of the individual.  Defaults to None.
		
		-``crossover``: The crossover operator which the individual will be initialized with.  Defaults to None.
		
		-``fitness``:  The fitness value which the individual will be initialized with.  Defaults to largest negative int value.
		"""
		self.genes = genes
		self.fitness = fitness
		self.crossover = crossover
		self.id = id
		self.parents = (None, None)
	def distance(self, other):
		"""
		Returns the genetic distance between this individual and another.
		
		Parameters:
		
		-``other``: The other individual which this individual is being compared to with respect to genetic distance.
		"""
		return math.sqrt(sum([(mine - theirs) ** 2 for mine, theirs in zip(self.genes, other.genes)]))
	def getFit(self):
		"""
		Returns the fitness of the individual.
		"""
		return self.fitness
	def __cmp__(self, other):
		"""
		Returns 1 if this individual is more fit, 0 if there is no fitness difference, or -1 if ``other`` is more fit.
		
		Parameters:
		
		-``other``: The other individual which this individual is being compared to with respect to fitness.
		"""
		return 1 if self.fitness > other.fitness else -1 if self.fitness < other.fitness else 0
	def __str__(self):
		"""
		Returns the string value of the individual by printing the concatenation of the string values of each gene in the individual.
		"""
		return "(" + ",".join(map(str, self.genes)) + ") = " + str(self.fitness)
	def __repr__(self):
		"""
		Returns the string value of the individual by printing the concatenation of the string values of each gene in the individual.
		"""
		return str(self)
	
	
		
		
		
