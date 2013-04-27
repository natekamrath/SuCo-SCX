"""
This module contains the SCX crossover oprator as well as all other crossover operator algorithms
"""

import random
import Util
import copy
import Mutation

class Construct(object):
	"""
	Base class for SCX operator primitive parameters
	"""
	def getValue(self, prev, length):
		"""
		returns the value of the construct
		"""
		raise Exception("A construct did not override getValue")

class Number(Construct):
	"""
	Class for the ``Number`` construct
	"""
	def __init__(self):
		"""
		Creates a new ``Number`` construct
		"""
		self.value = random.random()
	
	def getValue(self, prev, length):
		"""
		Returns the value of the construct
		
		Parameters:
		
		-``prev``: Previous construct. Unused in the ``Number`` construct
		
		-``length``: Length of the construct. Unused in the ``Number`` construct
		"""
		return int(length * self.value)
	def __str__(self):
		"""
		Returns the value of the construct as a string
		"""
		return str(self.value)

class Random(Construct):
	"""
	Class for the ``Random`` construct
	"""
	def getValue(self, prev, length):
		"""
		Returns the value of the construct
		
		Parameters:
		
		-``prev``: Previous construct. Unused in the ``Number`` construct
		
		-``length``: Length of the construct. Unused in the ``Number`` construct
		"""
		return int(length * random.random())
	def __str__(self):
		"""
		Returns the value of the construct as a string
		"""
		return "RAND"

class Inline(Construct):
	"""
	Class for the Inline construct
	"""
	def getValue(self, prev, length):
		"""
		Returns the value of the construct
		
		Parameters:
		
		-``prev``: The previous construct
		
		-``length``: The length of the Inline operation
		"""
		if prev is None:
			return None
		return (prev + (length / 2)) % length
	def __str__(self):
		"""
		Returns the value of the construct as a string
		"""
		return "INLINE"

class Primitive(object):
	"""
	Class for the Primitive object.  This is the base class for SCX primitives.
	"""
	def __init__(self):
		"""
		Creates a new primitive instance
		"""
		constructTypes = Util.childClasses(Construct)
		constructTypes.remove(Inline)
		self.start = random.choice(constructTypes + [Inline])()
		self.end = random.choice(constructTypes + [Inline])()
		self.special = random.choice(constructTypes)()

	def getValues(self, genes):
		"""
		Returns the value start and end genes
		
		Parameters:
		
		-``genes``: List of genes which the primitive is performing the crossover on.
		"""
		length = len(genes)
		start = self.start.getValue(None, length)
		end = self.end.getValue(start, length)
		if start is None:
			if end is None:
				# Both Inline
				start = end = random.randint(0, length - 1)
			else:
				start = self.start.getValue(end, length)
		return (start, end)

	def execute(self, genes):
		"""
		Performs the primitive function on the specified genes. Should be overridden by child class.
		"""
		raise Exception("Primitive did not override execute")

	def weightedAverage(self, x, y, weight):
		"""
		Returns the weighted average of the two values
		
		Parameters:
		
		-``x``: The first value (usually of a primitive).
		
		-``y``: The second value (usually of a primitive).
		
		-``weight``: The weight for the weighted average computation.
		"""
		return x * weight + y * (1 - weight)

	def __str__(self):
		"""
		Returns the value of the primitive as a string
		"""
		return str(type(self)) + "Start:%s, End:%s, Special:%s" % (self.start, self.end, self.special)

class Swap(Primitive):
	"""
	Class for the swap primitive.  This primitive functions like the swap function of traditional crossover operators.
	"""
	def execute(self, genes):
		"""
		Performs a swap function on the specified genes
		
		Parameters:
		
		-``genes``: The genes which are to be swapped
		"""
		length = len(genes)
		start, end = self.getValues(genes)
		value = self.special.getValue(None, length) / float(length)
		width = int(length / random.uniform(2, length))

		startPiece = [genes[(start + i) % length] for i in range(width)]
		endPiece = [genes[(end + i) % length] for i in range(width)]

		for i in range(width):
			genes[(start + i) % length] = endPiece[i]
			genes[(end + i) % length] = startPiece[i]

class Merge(Primitive):
	"""
	Class for the Merge primitive.  This primitive performs a merge function similar to that of arithmetic crossover
	"""
	def execute(self, genes):
		start, end = self.getValues(genes)
		weight = self.special.getValue(None, 10000) / 10000.0
		genes[start], genes[end] = (self.weightedAverage(genes[start], genes[end], weight),
									self.weightedAverage(genes[end], genes[start], weight))
class Crossover(object):
	"""
	SCX class.  This class holds a crossover operator which can be self-adapted or evolved.
	"""
	def __init__(self, initialLength):
		"""
		Creates a new crossover instance.
		
		Parameters:
		
		-``initialLength: The initial length (or number of primitives) in the crossover individual.
		"""
		primitives = Util.childClasses(Primitive)
		self.genes = [random.choice(primitives)() for iterator in range(initialLength)]

	def execute(self, genes):
		"""
		Performs the crossover encoded in self on ``genes``.
		
		Parameters:
		
		-``genes``: the genes for the crossover operation.
		"""
		for primitive in self.genes:
			primitive.execute(genes)
		return genes

	def __str__(self):
		"""
		Returns the string representation of the crossover operator.
		"""
		ret = "Crossover{"
		for primitive in self.genes:
			ret += str(primitive)
		ret += "}"
		return ret

	def fixedPointReproduction(self, other, constants):
		"""
		Performs a fixed point reproduction with the crossover operator on the specified individuals.
		
		Parameters:
		
		-``other``: The other 
		"""
		child = Crossover(0)
		p1genes = random.randint(0, len(self.genes))
		p2genes = random.randint(0, len(other.genes))
		over = p1genes + p2genes - 200
		if over > 0:
			p1genes -= over / 2
			p2genes -= over / 2
			# Ensure non negative
			p1genes = max(0, p1genes)
			p2genes = max(0, p2genes)
		child.genes = self.genes[:p1genes] + other.genes[p2genes:]
		if len(child.genes) == 0:
			child.genes = [random.choice((self.genes[0], other.genes[-1]))]
		return child

	def variableReproduction(self, other, constants):
		child = Crossover(0)
		myStart, otherStart = random.randint(0, len(self.genes)), random.randint(0, len(other.genes))
		child.genes = [self.genes[(myStart + i) % len(self.genes)] for i in range(random.randint(0, len(self.genes)))] + \
			[other.genes[(otherStart + i) % len(other.genes)] for i in range(random.randint(0, len(other.genes)))]
		if len(child.genes) == 0:
			child.genes = [random.choice(self.genes + other.genes)]
		while len(child.genes) > constants["dimensions"] * 2:
			del child.genes[random.randint(0, len(child.genes) - 1)]
		return child

	def randomReproduction(self, other, constants):
		return Crossover(random.randint(1, min(len(self.genes) + len(other.genes), constants["dimensions"] * 2)))

	def randomFixedLengthReproduction(self, other, constants={}):
		return Crossover((len(self.genes) + len(other.genes)) / 2)

	def randomLengthReproduction(self, other, constants):
		return Crossover(random.randint(1, constants["dimensions"] * 2))

def uniform(child, parents, constants={}):
	"""
	Uniform crossover algorithm.

	Parameters:

	-``child``: The child being created by the crossover.

	-``parents``: List of parents which will be used to create the child.

	-``constants``: Config settings for the EA from the config files specified at run time
	"""
	maxCommon = len(parents[0].genes)
	child.genes = [random.choice([parent.genes[g] for parent in parents]) for g in range(maxCommon)]

def npoint(child, parents, constants):
	"""
	N-Point crossover algorithm.

	Parameters:

	-``child``: The child being created by the crossover.

	-``parents``: List of parents which will be used to create the child.

	-``constants``: Config settings for the EA from the config files specified at run time
	"""
	numberOfPoints = constants["numberOfPoints"]
	maxCommon = min(len(parent.genes) for parent in parents)
	points = set(random.sample(range(maxCommon), min(numberOfPoints, maxCommon)))
	parent = 0
	child.genes = []
	for g in range(maxCommon):
		if g in points: parent += 1
		child.genes.append(parents[parent % len(parents)].genes[g])

def arithmetic(child, parents, constants={}):
	"""
	Arithmetic crossover algorithm.

	Parameters:

	-``child``: The child being created by the crossover.

	-``parents``: List of parents which will be used to create the child.

	-``constants``: Config settings for the EA from the config files specified at run time
	"""
	child.genes = [float(sum(values)) / len(values) for values in zip(*[parent.genes for parent in parents])]

def scx(child, parents, constants={}):
	"""
	SCX crossover algorithm.  Performs the crossover specified by the parents' crossover operators.

	Parameters:

	-``child``: The child being created by the crossover.

	-``parents``: List of parents which will be used to create the child.

	-``constants``: Config settings for the EA from the config files specified at run time
	"""
	if parents[0].crossover != None and parents[1].crossover != None:
		method = Util.classMethods(Crossover)[constants["scxRecombination"]]
		child.crossover = method(parents[0].crossover, (parents[1].crossover), constants)
		mutation = Util.moduleFunctions(Mutation)[constants["scxMutation"]]
		mutation(child, constants)
		child.genes = child.crossover.execute(parents[0].genes + parents[1].genes)
		child.genes = child.genes[:len(child.genes) / 2]
	else:
		raise Exception("Trying to SCX when parent's crossover is None")
        
        
def scxFromSupport(child, parents, scx, constants={}):
	"""
	Performs the scx crossover with individuals from a SCX support population.

	Parameters:

	-``child``: The child being created by the crossover.

	-``parents``: List of parents which will be used to create the child.

	-``scx``: The scx operator from the support population.

	-``constants``: Config settings for the EA from the config files specified at run time
	"""
	if scx.crossover != None:
		child.genes = scx.crossover.execute(parents[0].genes + parents[1].genes)
		child.genes = child.genes[:len(child.genes) /2]
	else:
		raise Exception("SCX from support was None")
