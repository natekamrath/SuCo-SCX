"""
This module contains classes for the different Individual's gene types.
"""
import random

class GeneType(object):
	"""
	Base class for the gene types.
	"""
	def __init__(self, minimum, maximum):
		"""
		Creates a new instance of the gene type class.
		
		Parameters:
		
		-``minimum``: The minimum gene value allowed.
		
		-``maximum``: The maximum gene value allowed.
		"""
		self.min = minimum
		self.max = maximum
	def fix(self, genes):
		for g in range(len(genes)):
			if genes[g] < self.min:
				genes[g] = self.min
			if genes[g] > self.max:
				genes[g] = self.max
			genes[g] = self.fixGene(genes[g])
	def fixGene(self, gene): return gene
	def randomGenome(self, length): return [self.randomValue() for _ in range(length)]

class INT(GeneType):
	"""
	Class for gens with type integer
	"""
	def randomValue(self): return random.randint(self.min, self.max)
	"""
	Returns a random value between the gene's min and max values.
	"""
	def fixGene(self, gene):
		"""
		Fixes the gene if the value is not a valid INT gene type.
		
		Parameters:
		
		-``gene``: The gene to be fixed.
		"""
		value = int(gene)
		if value != gene and random.random() < gene - value:
			value += 1
		return value

class FLT(GeneType):
	"""
	Class for floating point gene types.
	"""
	def __init__(self, minimum, maximum, shouldFix = True):
		"""
		creates a new instance of the FLT class
		
		Parameters:
		
		-``minimum``: the minimum gene value allowed.
		
		-``maximum``: the maximum gene value allowed.
		
		-``shouldFix``: if gene values should be fixed.  Defaults to True.
		"""
		super(FLT, self).__init__(minimum, maximum)
		self.shouldFix = shouldFix
		
	def fix(self, genes):
		"""
		A function for fixing FLT genes
		
		Parameters:
		
		-``genes``: A list of genes to be fixed.
		"""
		if self.shouldFix:
			for g in range(len(genes)):
				if genes[g] < self.min:
					genes[g] = self.min
				if genes[g] > self.max:
					genes[g] = self.max
				genes[g] = self.fixGene(genes[g])
				
	def randomValue(self): 
		"""
		Returns a random value within the genes allowed min and max values.
		"""
		return random.uniform(self.min, self.max)

class MutationIndividualGene(GeneType):
	"""
	Class for mutation support individual gene type.
	"""
	def __init__(self, minimum, maximum, initialValue = 'random', shouldFix = True):
		"""
		creates a new instance of the mutation individual gene.
		
		Parameters:
		
		-``minimum``: the minimum gene value allowed.
		
		-``maximum``: the maximum gene value allowed.
		
		-``initialValue: the initial value to seed the gene with.  Defaults to random.
		
		-``shouldFix``: if gene values should be fixed.  Defaults to True.
		"""
		super(MutationIndividualGene, self).__init__(minimum, maximum)
		self.shouldFix = shouldFix
		self.initialValue = initialValue
		
	def fix(self, genes):
		"""
		A function for fixing FLT genes
		
		Parameters:
		
		-``genes``: A list of genes to be fixed.
		"""
		if self.shouldFix:
			for g in range(len(genes)):
				if genes[g] < self.min:
					genes[g] = self.min
				if genes[g] > self.max:
					genes[g] = self.max
				genes[g] = self.fixGene(genes[g])
		
	def randomValue(self):
		"""
		Returns a random value within the bounds of the gene's min and max values if the initial value is random.  Otherwise returns the initial value.
		"""
		if self.initialValue == 'random':
			return random.uniform(self.min, self.max)
		else:
			return self.initialValue
			


