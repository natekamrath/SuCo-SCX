"""
This module contains the classes for the different types of populations used in the experiments.
"""
import ParentSelection
import SurvivorSelection
import Crossover
import Mutation
import GeneTypes
import Util
import sys
import math
import random
from Individuals import Individual
from itertools import combinations
import inspect

class Population(object):
	"""
	Class for a basic EA population
	"""
	def __init__(self, constants):
		"""
		creates a new population instance.
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time
		"""
		self.geneType = Util.moduleClasses(GeneTypes)[constants["geneType"]](constants["min"], constants["max"])
		self.individuals = []
		try: self.parentSelection = Util.moduleFunctions(ParentSelection)[constants["parentSelection"]]
		except KeyError: pass
		try: self.crossover = Util.moduleFunctions(Crossover)[constants["crossover"]]
		except KeyError: pass
		try: self.mutation = Util.moduleFunctions(Mutation)[constants["mutation"]]
		except KeyError: pass
		try: self.survivorSelection = Util.moduleFunctions(SurvivorSelection)[constants["survivorSelection"]]
		except KeyError: pass
		if "unique" in constants and constants["unique"]:
			self.next = self.unique(constants).next
		else:
			self.next = self.generator(constants).next
		self.id = None

	def initialIndividuals(self, constants):
		"""
		Creates the initial population of random individuals.
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time
		"""
		for i in range(constants["popSize"]):
			if "initialCrossoverLength" in constants:
				cross = Crossover.Crossover(constants["initialCrossoverLength"])
			else:
				cross = None
			individual = Individual(self.geneType.randomGenome(constants["dimensions"]), i, cross)
			self.id = i
			yield individual
			self.individuals.append(individual)

	def generator(self, constants):
		"""
		Carries out the standard evolutionary process of parent selection, evolution, and survivor selection.
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time
		"""
		for individual in self.initialIndividuals(constants):
			yield individual

		while True:
			if constants["logAvgFitness"]:
				print self.getAvgFitness()
			parents = self.parentSelection(self.individuals, constants["offSize"] * constants["parentsPerChild"], constants)
			for i in range(0, len(parents), constants["parentsPerChild"]):
				family = parents[i:i + constants["parentsPerChild"]]
				child = Individual(id=id)
				child.parents = [parent.id for parent in family]
				self.id += 1
				self.crossover(child, family, constants)
				self.mutation(child, constants, family)
				self.geneType.fix(child.genes)
				yield child
				self.individuals.append(child)
			self.individuals = self.survivorSelection(self.individuals, constants["popSize"], constants)
            

class CoPopulation(Population):
	"""
	Base class for a support population.
	"""
	def evalPop(self, constants):
		"""
		Evaluates all the support individuals
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time
		"""
		# reset all of the individuals
		for individual in self.individuals:
			individual.fitness = 0
			individual.evalCount = 0
		# all individuals are valid to choose at the start
		valid = range(len(self.individuals))
		# keep going until no one is valid
		while(len(valid) > 0):
			choice = random.choice(valid)
			# if this individual needs more evaluations
			if self.individuals[choice].evalCount < constants["evalsPerGeneration"]:
				# store the current fitness
				fitness = self.individuals[choice].fitness
				yield self.individuals[choice]
			else:
				valid.remove(choice)

	def generator(self, constants):
		"""
		Copopulation generator.  Should be overridden"
		"""
		print "CoPopulation Generator not over ridden"
			
class MutationCoPopulation(CoPopulation):
	"""
	Class for mutation support population
	"""
	def __init__(self, constants):
		"""
		Creates a new mutation copopulation instance
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time
		"""
		super(MutationCoPopulation, self).__init__(constants)
		"""
		upperBound = abs(random.gauss(constants["mutationStepSize"], constants["mutationStepSize"]))
		lowerBound = abs(random.gauss(constants["mutationStepSize"], constants["mutationStepSize"]))
		if lowerBound > upperBound:
			temp = lowerBound
			lowerBound = upperBound
			upperBound = temp
		"""
		self.geneType = Util.moduleClasses(GeneTypes)["MutationIndividualGene"](0.00, 1, constants["mutationStepSize"])
		try:self.mutation = Util.moduleFunctions(Mutation)["gaussian"]
		except KeyError: print "ERROR: no mutation specified for mutation copopulation"
		
	def generator(self, constants):
		"""
		Carries out the standard evolutionary process of parent selection, evolution, and survivor selection.
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time
		"""
		for _ in self.initialIndividuals(constants):
			pass
		for individual in self.evalPop(constants):
			yield individual
		while True:
			parents = self.parentSelection(self.individuals, constants["offSize"] * constants["parentsPerChild"], constants)
			for i in range(0, len(parents), constants["parentsPerChild"]):
				family = parents[i:i + constants["parentsPerChild"]]
				child = Individual(id=id)
				child.parents = [parent.id for parent in family]
				self.id += 1
				self.crossover(child, family, constants)
				self.mutation(child, constants, family)
				self.geneType.fix(child.genes)
				self.individuals.append(child)
			for individual in self.evalPop(constants):
				yield individual
			self.individuals = self.survivorSelection(self.individuals, constants["popSize"], constants)
            
class SCXCoPopulation(CoPopulation):
	"""
	Class for the SCX operator support population.
	"""
	def __init__(self, constants):
		"""
		Creates a new scx copopulation instance.
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time
		"""
		super(SCXCoPopulation, self).__init__(constants)
		try: self.mutation = Util.moduleFunctions(Mutation)[constants["scxMutation"]]
		except KeyError: print "ERROR: no mutation specified for scxMutation"
		self.crossover = Util.classMethods(Crossover.Crossover)[constants["scxRecombination"]]
		
	def initialIndividuals(self, constants):
		"""
		Creates the initial random population.
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time
		"""
		for i in range(constants["popSize"]):
			if "initialCrossoverLength" in constants:
				cross = Crossover.Crossover(constants["initialCrossoverLength"])
			else:
				print "ERR: initial crossover length not specified for SCX support population"
			individual = Individual([], i, cross)
			self.id = i
			yield individual
			self.individuals.append(individual)
		
	def generator(self, constants):
		"""
		Carries out the standard evolutionary process of parent selection, evolution, and survivor selection.
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time
		"""
		for _ in self.initialIndividuals(constants):
			pass
		for individual in self.evalPop(constants):
			yield individual
		while True:
			parents = self.parentSelection(self.individuals, constants["offSize"] * constants["parentsPerChild"], constants)
			for i in range(0, len(parents), constants["parentsPerChild"]):
				family = parents[i:i + constants["parentsPerChild"]]
				child = Individual(id=id)
				child.parents = [parent.id for parent in family]
				self.id += 1
				child.crossover = self.crossover(family[0].crossover, family[1].crossover, constants)
				self.mutation(child, constants, family)
				self.individuals.append(child)
			for individual in self.evalPop(constants):
				yield individual
			self.individuals = self.survivorSelection(self.individuals, constants["popSize"], constants)

class Primary(Population):
	"""
	Class for the primary individual populations
	"""
	def __init__(self, constants):
		"""
		Creates a new instance of the primary individual popoulation class
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time
		"""
		super(Primary, self).__init__(constants)
		try: self.mutation = Util.moduleFunctions(Mutation)["gaussRates"]
		except KeyError: print "ERROR: no mutation specified for scxMutation"
		
	def generator(self, constants, supports=None):
		"""
		Carries out the standard evolutionary process of parent selection, evolution, and survivor selection.
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time
		"""
		supportIndividual = None
		for individual in self.initialIndividuals(constants):
			if "MutationCoPopulation" not in constants["supportPopulations"]:
				# TODO Explain ranging 
				SAfix = GeneTypes.FLT(0.001, constants["mutationStepSize"])
				individual.stepSizes = SAfix.randomGenome(constants["dimensions"])
			yield individual, None

		while True:
			if constants["logAvgFitness"]:
				print self.getAvgFitness()
			#print "generation"
			parents = self.parentSelection(self.individuals, constants["offSize"] * constants["parentsPerChild"], constants)
			for i in range(0, len(parents), constants["parentsPerChild"]):
				family = parents[i:i + constants["parentsPerChild"]]

				child = Individual(id=id)
				child.parents = [parent for parent in family]
				self.id += 1
				#self.crossover(child, family, constants)
				supportIndividuals = list()
				if "MutationCoPopulation" in constants["supportPopulations"]:
					mutationSupportIndividual = supports[MutationCoPopulation]() #mutationSupport()
					rates = mutationSupportIndividual.genes
					supportIndividuals.append(mutationSupportIndividual)
					"""
					for r in rates:
						if r != constants["mutationStepSize"]:
							print "rates changed after creation"
					"""
					#print "mutation created"
				elif "MutationCoPopulation" not in constants["supportPopulations"]:
					bias = 1 / math.sqrt(2.0 * constants["dimensions"]) * random.gauss(0, 1)
					tau = 1 / math.sqrt(2.0 * math.sqrt(constants["dimensions"]))
					child.stepSizes = [(sum(psteps) / len(psteps)) * math.exp(bias + tau * random.gauss(0, 1)) for psteps in zip(*[f.stepSizes for f in family])]
					SAfix.fix(child.stepSizes)
					rates = child.stepSizes
				
				if "SCXCoPopulation" in constants["supportPopulations"]:
					crossoverSupportIndividual = supports[SCXCoPopulation]() #crossoverSupport()
					#print len(crossoverSupportIndividual.crossover.genes)
					Crossover.scxFromSupport(child, parents, crossoverSupportIndividual, constants)
					#print len(crossoverSupportIndividual.crossover.genes)
					supportIndividuals.append(crossoverSupportIndividual)
					#print "crossover created"
					
				elif "SCXCoPopulation" not in constants["supportPopulations"]:
					self.crossover(child, family, constants)
					#print len(child.crossover.genes)
				
				if constants["SuCoLevel"] == "Static":
					rates = constants["stepSizes"]
					
				"""
				for r in rates:
					if r != constants["mutationStepSize"]:
						print "rates changed"
				"""
						
				self.mutation(child, constants, rates)
				self.geneType.fix(child.genes)
				yield child, supportIndividuals
				self.individuals.append(child)

			self.individuals = self.survivorSelection(self.individuals, constants["popSize"], constants)

class SuCo(Population):
	"""
	Class for the Supportive Coevoution Population as a whole.  Contains all Primary and Support Populations
	"""
	def generator(self, constants):
		"""
		Carries out the standard evolutionary process of parent selection, evolution, and survivor selection.
		
		Parameters:
		
		-``constants``: Config settings for the EA from the config files specified at run time
		"""
		if constants["SuCoLevel"] == "Support":
			supportConstants = dict(constants)
			supportConstants["max"] = constants["mutationStepSize"]
			supportConstants["min"] = 0.001
			supportConstants["mutation"] = "tau"
			for key, value in constants.iteritems():
				if 'support_' in key:
					supportConstants[key.lstrip("support_")] = value
					
			popTypes = [dict(inspect.getmembers(sys.modules[__name__], inspect.isclass))[p] for p in constants["supportPopulations"]]
			supports = dict()
			if "joined" in constants["SuCoFitnessDecorators"]:
				for pop in popTypes:
					supports[pop] = [pop(supportConstants).generator(supportConstants).next] * constants["parallelPopulations"]
					
			else:
				for pop in popTypes:
					supports[pop] = [pop(supportConstants).generator(supportConstants).next for _ in range(constants["parallelPopulations"])]
			
		primaries = list()
		for i in range(constants["parallelPopulations"]):
			popsForPrimary = dict()
			for pop in popTypes:
				popsForPrimary[pop] = (supports[pop][i])
			primaries.append(Primary(constants).generator(constants, popsForPrimary).next)
		
		counter = 0
		while True:
			counter +=1
			for p in primaries:
				child, supportIndividuals = p()
				yield child
				family = child.parents
				distScale = None
				if constants["SuCoLevel"] == "Support" and supportIndividuals is not None:
					fitness = child.fitness
					if "relative" in constants["SuCoFitnessDecorators"]:
						fitness -= (sum([p.fitness for p in family]) / len(family))
					if "distance" in constants["SuCoFitnessDecorators"]:
						pdist = family[0].distance(family[1])
						if pdist < 1: pdist = 1
						if pdist != 0:
							distScale = (child.distance(family[0]) * child.distance(family[1])) / pdist
							
					temp = child.fitness - ((family[0].fitness + family[1].fitness) / 2)
					for supportIndividual in supportIndividuals:
						supportIndividual.fitness += temp * distScale if distScale != None else temp
						try: supportIndividual.evalCount += 1
						except: supportIndividual.evalCount = 1
