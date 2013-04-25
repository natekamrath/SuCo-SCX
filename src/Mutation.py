import random
import Crossover
import Util
import copy
import math
def bitFlip(child, constants, parents=None):
    rate = constants["mutationRate"]
    for g in range(len(child.genes)):
        if random.random() < rate:
            child.genes[g] = Util.flip(child.genes[g])

def creep(child, constants, parents=None):
    step = constants["mutationStepSize"]
    rate = constants["mutationRate"]
    child.genes = [g + random.normalvariate(0, step) if random.random() < rate else g for g in child.genes]

def gaussian(child, constants, parents=None):
    step = constants["mutationStepSize"]
    rate = constants["mutationRate"]
    for g in range(len(child.genes)):
		if random.random() < rate: #comment this part out and mutate every gene performance offset rast much better
			child.genes[g] += random.gauss(0, step)

def scxOriginal(child, constants, parents=None):
    rate = constants["mutationRate"] * constants['dimensions'] / len(child.crossover.genes)
    for g in range(len(child.crossover.genes)):
        if random.random() < rate:
            child.crossover.genes[g] = random.choice(Util.childClasses(Crossover.Primitive))()

def scxRate(child, constants, parents=None):
    rate = constants["scxMutationRate"]
    for g in range(len(child.crossover.genes)):
        if random.random() < rate:
            child.crossover.genes[g] = random.choice(Util.childClasses(Crossover.Primitive))()

def scxLengthIndependent(child, constants, parents=None):
    rate = constants["mutationRate"]
    for g in range(len(child.crossover.genes)):
        if random.random() < rate:
            child.crossover.genes[g] = random.choice(Util.childClasses(Crossover.Primitive))()

def scxMinor(child, constants, parents=None):
    rate = constants["mutationRate"]
    for g in range(len(child.crossover.genes)):
        if random.random() < rate:
            child.crossover.genes[g] = copy.deepcopy(child.crossover.genes[g])
            choice = random.choice([0, 1, 2])
            constructTypes = Util.childClasses(Crossover.Construct)
            constructTypes.remove(Crossover.Inline)
            if choice == 0:
                child.crossover.genes[g].start = random.choice(constructTypes)()
            elif choice == 1:
                child.crossover.genes[g].end = random.choice(constructTypes + [Crossover.Inline])()
            else:
                child.crossover.genes[g].special = random.choice(constructTypes)()

def gaussRates(child, constants, stepSizes):
	rate = constants["mutationRate"]
	for g in range(len(child.genes)):
		if random.random() < rate: #added
			child.genes[g] += random.gauss(0, stepSizes[g])

def tau(child, constants, parents=None):
    bias = 1 / math.sqrt(2.0 * constants["dimensions"]) * random.gauss(0, 1)
    tau = 1 / math.sqrt(2.0 * math.sqrt(constants["dimensions"]))
    child.genes = [gene * math.exp(bias + tau * random.gauss(0, 1)) for gene in child.genes]

def revac(child, constants, parents):
    step = constants["mutationStepSize"]
    data = []
    for g in range(len(child.genes)):
        alleles = sorted([individual.genes[g] for individual in parents])
        data.append(alleles[-1] - alleles[0])
        middle = alleles.index(child.genes[g])

        # finds the range of the mutation
        bottom = alleles[middle - step] if middle - step >= 0 else 2 * alleles[middle] - alleles[middle + step]
        top = alleles[middle + step] if middle + step < len(alleles) else 2 * alleles[middle] - alleles[middle - step]
        child.genes[g] = random.uniform(bottom, top)
    print 'Genetic Width', data
