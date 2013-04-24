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
	def __init__(self, constants):
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

	def unique(self, constants):
		self.seen = {}
		real, fake = 0, 0
		for possible in self.generator(constants):
			key = possible.__hash__()
			try:
				possible.fitness = self.seen[key]
				fake += 1
				#if fake > 10000:
				#    print "STALLED", fake, real
				#    break
				#raise KeyError()
			except KeyError:
				real += 1
				yield possible
				self.seen[key] = possible.fitness

	def initialIndividuals(self, constants):
		for i in range(constants["popSize"]):
			if "initialCrossoverLength" in constants:
				cross = Crossover.Crossover(constants["initialCrossoverLength"])
			else:
				cross = None
			individual = Individual(self.geneType.randomGenome(constants["dimensions"]), i, cross)
			self.id = i
			yield individual
			self.individuals.append(individual)

	def localSearch(self, index):
		individual = self.individuals[index]
		improved = True
		while improved:
			best = individual
			improved = False
			indexes = range(len(individual.genes))
			random.shuffle(indexes)
			for g in indexes:
				flipped = Individual([x for x in individual.genes], self.id, 0)
				self.id += 1
				flipped.genes[g] = Util.flip(flipped.genes[g])
				yield flipped
				if best < flipped:
					best = flipped
					if self.constants["quick"]:
						individual = best
					improved = True
			self.localSearchSteps += 1
			individual = best
		self.individuals[index] = individual


	def generator(self, constants):
		for individual in self.initialIndividuals(constants):
			yield individual

		if "localSearch" in constants and constants["localSearch"]:
			# performs local search on all of the individuals
			for i in range(constants["popSize"]):
				for individual in self.localSearch(i):
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
			
	def getAvgFitness(self):
		total = 0
		for individual in self.individuals:
			total += individual.fitness
		return total/len(self.individuals)
            

class CoPopulation(Population):
    def evalPop(self, constants):
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
        print "CoPopulation Generator not over ridden"
            
class MutationCoPopulation(CoPopulation):
	def __init__(self, constants):
		super(MutationCoPopulation, self).__init__(constants)
		#self.geneType = Util.moduleClasses(GeneTypes)["FLT"](0.00, 1)
		#self.geneType = Util.moduleClasses(GeneTypes)["FLT"](constants["mutationStepSize"] - .01, constants["mutationStepSize"]+.01)
		try:self.mutation = Util.moduleFunctions(Mutation)["tau"]
		except KeyError: print "ERROR: no mutation specified for mutation copopulation"
	def generator(self, constants):
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
			#print "evolving mutations: " + str(indsProduced)
            
class SCXCoPopulation(CoPopulation):
	def __init__(self, constants):
		super(SCXCoPopulation, self).__init__(constants)
		try: self.mutation = Util.moduleFunctions(Mutation)[constants["scxMutation"]]
		except KeyError: print "ERROR: no mutation specified for scxMutation"
		self.crossover = Util.classMethods(Crossover.Crossover)[constants["scxRecombination"]]
		
	def initialIndividuals(self, constants):
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
		for _ in self.initialIndividuals(constants):
			pass
		for individual in self.evalPop(constants):
			yield individual
		#print "individuals: " + str(len(self.individuals))
		while True:
			parents = self.parentSelection(self.individuals, constants["offSize"] * constants["parentsPerChild"], constants)
			#print "parents: " + str(len(parents))
			for i in range(0, len(parents), constants["parentsPerChild"]):
				family = parents[i:i + constants["parentsPerChild"]]
				child = Individual(id=id)
				child.parents = [parent.id for parent in family]
				self.id += 1
				child.crossover = self.crossover(family[0].crossover, family[1].crossover, constants)
				#child.crossover = parents[0].crossover.fixedPointReproduction(parents[1].crossover, constants)
				#print len(child.crossover.genes)
				self.mutation(child, constants, family)
				self.individuals.append(child)
			for individual in self.evalPop(constants):
				yield individual
			self.individuals = self.survivorSelection(self.individuals, constants["popSize"], constants)
			#print "evolving crossovers"

class Primary(Population):	
	def generator(self, constants, supports=None):
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
				self.mutation(child, constants, rates)
				self.geneType.fix(child.genes)
				yield child, supportIndividuals
				self.individuals.append(child)

			self.individuals = self.survivorSelection(self.individuals, constants["popSize"], constants)

class SuCo(Population):
	def generator(self, constants):
		
		if constants["SuCoLevel"] == "Support":
			supportConstants = dict(constants)
			supportConstants["max"] = constants["mutationStepSize"]
			supportConstants["min"] = 0.001
			supportConstants["mutation"] = "tau"
			for key, value in constants.iteritems():
				if 'support_' in key:
					supportConstants[key.lstrip("support_")] = value
					
			"""
			popType = CoPopulation if "reeval" in constants["SuCoFitnessDecorators"] else Population
			if "joined" in constants["SuCoFitnessDecorators"]:
				support = [popType(supportConstants).generator(supportConstants).next] * constants["parallelPopulations"]
			else:
				support = [popType(supportConstants).generator(supportConstants).next for _ in range(constants["parallelPopulations"])]
		else:
			support = [None] * constants["parallelPopulations"]
			"""
			popTypes = [dict(inspect.getmembers(sys.modules[__name__], inspect.isclass))[p] for p in constants["supportPopulations"]]
			#print popTypes
			supports = dict()
			if "joined" in constants["SuCoFitnessDecorators"]:
				for pop in popTypes:
					supports[pop] = [pop(supportConstants).generator(supportConstants).next] * constants["parallelPopulations"]
					
			else:
				for pop in popTypes:
					supports[pop] = [pop(supportConstants).generator(supportConstants).next for _ in range(constants["parallelPopulations"])]
			
		primaries = list()
		#primaries = [Primary(constants).generator(constants, supportPop).next for supportPop in support]
		for i in range(constants["parallelPopulations"]):
			popsForPrimary = dict()
			for pop in popTypes:
				popsForPrimary[pop] = (supports[pop][i])
			primaries.append(Primary(constants).generator(constants, popsForPrimary).next)
		
		counter = 0
		#print str(len(primaries)) + " = number primary populations"
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
							
					#temp = (abs(family[0].fitness + family[1].fitness) / 2) - abs(child.fitness)
					temp = child.fitness - ((family[0].fitness + family[1].fitness) / 2)
					for supportIndividual in supportIndividuals:
						supportIndividual.fitness += temp * distScale if distScale != None else temp # run on right and rosenbrock top left
						try: supportIndividual.evalCount += 1
						except: supportIndividual.evalCount = 1

class LTGA(Population):
    def __init__(self, constants):
        super(LTGA, self).__init__(constants)
        self.clusters = [(i,) for i in range(constants["dimensions"])]
        self.subtrees = [(i,) for i in range(constants["dimensions"])]
        self.constants = constants
        self.entropyLookup = {}
        self.mutants = {}
        self.crosses = {}

    def mergeClosestClusters(self, constants):
        #find which two clusters are the closest together
        if constants["pairwiseDistance"]:
            distance = self.pairwiseDistance
        else:
            distance = self.clusterDistance

        key = lambda C: distance(*C)
        lowest = min(map(key, combinations(self.clusters, 2)))
        minimums = filter(lambda C: key(C) == lowest, combinations(self.clusters, 2))
        # TODO Consider mentioning this in report
        c1, c2 = random.choice(minimums)
        self.clusters.remove(c1)
        self.clusters.remove(c2)
        self.clusters.append(c1 + c2)
        # Only add it as a subtree if it is not the root
        if len(self.clusters) != 1:
            self.subtrees.append((c1 + c2))

    def entropy(self, genes):
        occurances = {}
        for individual in self.individuals:
            key = tuple(individual.genes[g] for g in genes)
            try: occurances[key] += 1
            except KeyError: occurances[key] = 1
        total = float(len(self.individuals))
        return -sum(x / total * math.log(x / total, 2) for x in occurances.itervalues())

    def clusterDistance(self, c1, c2):
        try: return self.entropyLookup[(c1, c2)]
        except KeyError:
            self.entropyLookup[(c1, c2)] = 2 - (self.entropy(c1) + self.entropy(c2)) / max(self.entropy(c1 + c2), 0.000001)
            self.entropyLookup[(c2, c1)] = self.entropyLookup[(c1, c2)]
            return self.entropyLookup[(c1, c2)]

    def pairwiseDistance(self, c1, c2):
        try: return self.entropyLookup[(c1, c2)]
        except KeyError:
            self.entropyLookup[(c1, c2)] = sum(self.clusterDistance((a,), (b,)) for a in c1 for b in c2) / float(len(c1) * len(c2))
            return self.entropyLookup[(c1, c2)]

    def applyMask(self, p1, p2, mask):
        return Individual([p2.genes[g] if g in mask else p1.genes[g] for g in range(len(p1.genes))])

    def maskEqual(self, p1, p2, mask):
        for g in mask:
            if p1.genes[g] != p2.genes[g]:
                return False
        return True

    def updateUsed(self, genes):
        if self.constants["novelMutation"]:
            for mask in self.subtrees:
                key = tuple(genes[g] for g in mask)
                self.used[mask].append(key)

    def SHC(self, p1index):
        child = Individual(list(self.individuals[p1index].genes), fitness=self.individuals[p1index].fitness)
        for mask in self.subtrees:
            next = Individual(list(child.genes), fitness=child.fitness)
            key = tuple(next.genes[g] for g in mask)
            options = filter(lambda X: X != key, self.used[mask])
            if len(options) > 0 and "stochastic" in self.constants:
                options = [random.choice(options)]
            for option in set(options):
                for i, g in enumerate(mask):
                    next.genes[g] = option[i]
                yield next
                if child < next:
                    child = next
        #print [c - p if c != p else ' ' for c, p in zip(child.genes, self.individuals[p1index].genes)]
        self.individuals.append(child)


    def mutate(self, p1, mask):
        counter = 0
        bits = len(mask)
        child = Individual(list(p1.genes))
        value = self.used[mask]
        key = tuple(child.genes[g] for g in mask)
        safe = filter(lambda X: X != key, value)
        if len(safe) > 0:
            mutant = random.choice(safe)
        else:
            return child
        for i, g in enumerate(mask):
            child.genes[g] = mutant[i]
        return child

    def cross(self, p1Index, p2Index):
        p1, p2 = self.individuals[p1Index], self.individuals[p2Index]
        best = max(p1, p2)
        for mask in self.subtrees:
            new1, new2 = 0, 0
            #if self.constants["novelMutation"]:
            if self.maskEqual(p1, p2, mask):
                if self.constants["novelMutation"]:
                    c1 = self.mutate(p1, mask)
                    c2 = self.mutate(p2, mask)
                    yield c1
                    yield c2
                    if p1 < c1:
                        p1 = c1
                    if p2 < c2:
                        p2 = c2
                    if best < max(c1, c2):
                        best = max(c1, c2)
                    self.individuals[p1Index], self.individuals[p2Index] = p1, p2
            else:
                c1 = self.applyMask(p1, p2, mask)
                c2 = self.applyMask(p2, p1, mask)
                yield c1
                yield c2

                if max(p1, p2) < max(c1, c2):
                    p1, p2 = c1, c2
                    self.individuals[p1Index], self.individuals[p2Index] = p1, p2
                if best < max(c1, c2):
                    best = max(c1, c2)
        self.individuals.append(best)

    def generator(self, constants):
        self.restart = False
        try:
            problemKey = "%(problem)s_%(dimensions)i_%(trapSize)i_%(popSize)i_%(runNumber)i" % constants
        except KeyError:
            problemKey = "%(problem)s_%(dimensions)i_%(k)i_%(popSize)i_%(runNumber)i" % constants
        extra = "nolocal/" if not constants["localSearch"] else ""
        problemFile = constants["startPopFolder"] + extra + problemKey
        print problemKey
        try:
            stored = Util.loadConfiguration(problemFile)
            for blob in stored["pop"]:
                self.individuals.append(Individual(blob[0], fitness=blob[1]))
            self.localSearchSteps = stored["steps"]
            self.base = stored["evals"]
        except (IOError, ValueError):
            # evaluates all of the initial individuals
            for individual in self.initialIndividuals(constants):
                yield individual
            self.localSearchSteps = 0
            if constants["localSearch"]:
                # performs local search on all of the individuals
                for i in range(constants["popSize"]):
                    for individual in self.localSearch(i):
                        yield individual
            self.base = len(self.seen)

            store = {"pop":[(individual.genes, individual.fitness) for individual in self.individuals],
                     "steps":self.localSearchSteps,
                     "evals":self.base}
            Util.saveConfiguration(problemFile, store)
            self.restart = True
            return

        beforeGen = set(self.individuals)
        try:
            trapCount = self.constants["dimensions"] / self.constants["trapSize"]
            complete = dict((key, 0) for key in range(trapCount))
            for individual in self.individuals:
                for i in range(0, self.constants["dimensions"], self.constants["trapSize"]):
                    if sum(individual.genes[i:i + self.constants["trapSize"]]) == self.constants["trapSize"]:
                        complete[i / self.constants["trapSize"]] += 1
            self.mintrap = min(complete.items(), key=lambda X: X[1])[1]
        except:
            self.mintrap = 0
        self.used = {}
        while True:
            self.used = {}
            self.clusters = [(i,) for i in range(constants["dimensions"])]
            self.subtrees = [(i,) for i in range(constants["dimensions"])]
            random.shuffle(self.clusters)
            random.shuffle(self.subtrees)
            self.entropyLookup = {}
            while len(self.clusters) > 1:
                self.mergeClosestClusters(constants)
            if constants["smallestFirst"]:
                self.subtrees.sort(key=lambda tree: len(tree))
            else:
                self.subtrees.reverse()
            loops = [0, 1] if "always" not in self.constants else [0]
            for _ in loops:
                # Shuffle the population
                random.shuffle(self.individuals)
                if "novelMutation" in self.constants:
                    self.used = dict((mask, []) for mask in self.subtrees)
                    for individual in self.individuals:
                        self.updateUsed(individual.genes)

                for i in range(0, constants["popSize"], 2):
                    if "always" in self.constants:
                        for step in self.SHC(i):
                            yield step
                        for step in self.SHC(i + 1):
                            yield step
                    else:
                        for step in self.cross(i, i + 1):
                            yield step
            self.individuals = self.individuals[constants["popSize"]:]
            stagnat = set(self.individuals)
            if len(stagnat) == 1 or stagnat == beforeGen:
                break
            else:
                beforeGen = stagnat

