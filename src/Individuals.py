import sys
import math

class Individual(object):
    def __init__(self, genes=[], id=None, crossover=None, fitness= -sys.maxint - 1):
        self.genes = genes
        self.fitness = fitness
        self.crossover = crossover
        self.id = id
        self.parents = (None, None)
    def distance(self, other):
        return math.sqrt(sum([(mine - theirs) ** 2 for mine, theirs in zip(self.genes, other.genes)]))
    def getFit(self):
        return self.fitness
    def __cmp__(self, other):
        return 1 if self.fitness > other.fitness else -1 if self.fitness < other.fitness else 0
    def __str__(self):
        return "(" + ",".join(map(str, self.genes)) + ") = " + str(self.fitness)
    def __repr__(self):
        return str(self)
    def __hash__(self):
        return int("".join(map(str, self.genes)), 2)

class SCXIndividual(object):
	def __init__(self, crossover = None, fitness = -sys.maxint -1):
		self.crossover = crossover
		self.fitness = fitness
	
	
		
		
		
