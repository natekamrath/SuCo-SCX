import math
import random
import Population
import sys
import Experiments
import Util
import operator
import copy
class FitnessFunction(object):
    def __init__(self, constants={}):
        self.constants = constants

    def eval(self, child):
        raise Exception("Fitness function did not override eval")

def climb(f, x, step, lower, upper, limit=200):
    limit -= 1
    if limit < 0:
        return f(x)
    next = max((f(n), n) for n in [x, x - step, x + step] if lower <= n <= upper)[1]
    if next == x:
        return climb(f, x, step / 2, lower, upper, limit)
    else:
        return climb(f, next, step, lower, upper, limit)

class Rastrigin(FitnessFunction):
    def __init__(self, constants):
        self.A = constants["scalar"]
    def eval(self, child):
        child.fitness = round(-sum(map(self.function, child.genes)), 6)
    def function(self, x):
        return self.A + x * x - self.A * math.cos(2 * math.pi * x)

class ShiftedRastrigin(Rastrigin):
    def __init__(self, constants):
        self.A = constants["scalar"]
        rngstate = random.getstate()
        number = constants['problemSeed'] + constants["runNumber"]
        random.seed(number)
        self.offset = [random.uniform(constants["min"], constants["max"]) for _ in range(constants["dimensions"])]
        random.seed(rngstate)

    def eval(self, child):
        z = [x + y for x, y in zip(self.offset, child.genes)]
        child.fitness = round(-sum(map(self.function, z)), 6)

class OffsetRastrigin(Rastrigin):
    def __init__(self, constants):
        self.A = constants["scalar"]
        rngstate = random.getstate()
        number = constants['problemSeed'] + constants["runNumber"]
        random.seed(number)
        self.offset = [random.randint(0, 10) * 0.5 - 2.5 for _ in range(constants["dimensions"])]
        random.seed(rngstate)

    def eval(self, child):
        z = [x + y for x, y in zip(self.offset, child.genes)]
        child.fitness = round(-sum(map(self.function, z)), 6)


class Griewank(FitnessFunction):
    def __init__(self, constants):
        self.A = constants["scalar"]
    def eval(self, child):
        summation = sum([x ** 2 for x in child.genes])
        product = reduce(operator.mul, [math.cos(x / math.sqrt(i + 1)) for i, x in enumerate(child.genes)])
        child.fitness = 1 + 1 / 4000.0 * summation - product
        child.fitness = -round(child.fitness, 6)

class Rosenbrock(FitnessFunction):
    def __init__(self, constants):
        self.A = constants["scalar"]

    def function(self, genes, i):
        return (1 - genes[i]) ** 2 + self.A * (genes[i + 1] - genes[i] ** 2) ** 2

    def eval(self, child):
        child.fitness = 0
        for i in range(len(child.genes) - 1):
            child.fitness -= self.function(child.genes, i)
        child.fitness = round(child.fitness, 6)

class ShiftedRosenbrock(Rosenbrock):
    def __init__(self, constants):
        self.A = constants["scalar"]
        rngstate = random.getstate()
        number = constants['problemSeed'] + constants["runNumber"]
        random.seed(number)
        self.offset = [random.uniform(1 - constants["max"], 1 - constants["min"]) for _ in range(constants["dimensions"])]
        random.seed(rngstate)

    def eval(self, child):
        child.fitness = 0
        z = [x + y for x, y in zip(self.offset, child.genes)]
        for i in range(len(child.genes) - 1):
            child.fitness -= self.function(z, i)
        child.fitness = round(child.fitness, 6)


class OneMax(FitnessFunction):
    def eval(self, child):
        child.fitness = sum(child.genes) / float(len(child.genes))

class DTrap(FitnessFunction):
    def eval(self, child):
        child.fitness = 0
        trapSize = self.constants["trapSize"]
        for i in range(0, len(child.genes), trapSize):
            trap = sum(child.genes[i:i + trapSize])
            child.fitness += trapSize if trap == trapSize else trapSize - trap - 1
        child.fitness /= float(len(child.genes))

class MaxSpaceDTrap(FitnessFunction):
    def eval(self, child):
        child.fitness = 0
        trapSize = self.constants['trapSize']
        step = len(child.genes) / trapSize
        for i in range(0, step):
            trap = sum(child.genes[i::step])
            child.fitness += trapSize if trap == trapSize else trapSize - trap - 1
        child.fitness /= float(len(child.genes))

class JumpK(FitnessFunction):
    def __init__(self, constants):
        self.k = constants['k']
    def eval(self, child):
        count = sum(child.genes)
        n = len(child.genes)
        child.fitness = (self.k + count if count <= n - self.k or count == n else n - count) / float(len(child.genes))

class StepTrap(FitnessFunction):
    def stepify(self, trap):
        trapSize = self.constants["trapSize"]
        stepSize = self.constants["stepSize"]
        offset = trapSize % stepSize
        # return 0 if trap < stepSize - 1 else trap - (trap - offset) % stepSize
        return (trap + offset) / 2

    def eval(self, child):
        trapSize = self.constants["trapSize"]
        child.fitness = 0
        for i in range(0, len(child.genes), trapSize):
            # trap = sum(child.genes[i:i + trapSize])
            # fit = 0 if trap < stepSize - 1 else trap - (trap - odd) % stepSize
            fit = self.stepify(sum(child.genes[i:i + trapSize]))
            # print child.genes[i:i + trapSize], fit
            child.fitness += fit
        child.fitness /= float(len(child.genes))

class DeceptiveStepTrap(StepTrap):
    def eval(self, child):
        child.fitness = 0
        trapSize = self.constants["trapSize"]
        stepSize = self.constants["stepSize"]
        offset = (stepSize - trapSize) % stepSize
        for i in range(0, len(child.genes), trapSize):
            trap = sum(child.genes[i:i + trapSize])
            dtrap = trapSize if trap == trapSize else trapSize - trap - 1
            # print child.genes[i:i + trapSize], (dtrap + offset) / 2
            child.fitness += (dtrap + offset) / 2
        possible = math.ceil(float(trapSize) / stepSize) * (len(child.genes) / trapSize)
        child.fitness /= possible


class NK(FitnessFunction):
    answers = {}
    def buildEpistasis(self):
        indexes = range(self.n)
        self.epistasis = [random.sample(indexes, self.k + 1) for g in indexes]
        for i, row in enumerate(self.epistasis):
            if i not in row:
                row[0] = i
    def updateAnswers(self, newInfo):
        for key, value in newInfo.iteritems():
            if key not in NK.answers:
                NK.answers[key] = value
            else:
                NK.answers[key] = (min(NK.answers[key][0], value[0]), max(NK.answers[key][1], value[1]))

    def __init__(self, constants):
        self.n = constants["dimensions"]
        self.k = constants["k"]
        self.updateAnswers(Util.loadConfiguration(constants["nkSolutions"]))
        rngstate = random.getstate()
        problemNumber = constants['problemSeed'] + constants["runNumber"]
        random.seed(problemNumber)
        self.key = "%s-N:%i-K:%i-%i" % (str(type(self)), self.n, self.k, problemNumber)
        indexes = range(self.n)
        self.fitness = [[random.random() for _ in range(2 ** (self.k + 1))] for _ in indexes]
        self.buildEpistasis()
        self.solutions = constants['nkSolutions']
        self.range = (self.solve(min), self.solve(max))
        if self.key not in self.answers:
            self.answers[self.key] = self.range
            Util.saveConfiguration(constants["nkSolutions"], NK.answers)
        '''
        try:
            self.range = self.answers[self.key]
        except KeyError:
            print "Finding solution to new problem:", self.key
            self.range = (self.solve(min), self.solve(max))
            print self.range
            #self.answers[key] = self.range
            #NK.answers.update(Experiments.loadConfiguration(constants["nkSolutions"]))
            #Experiments.saveConfiguration(constants["nkSolutions"], NK.answers)
        '''
        '''
        try:
            self.range = self.answers[key]
        except KeyError:
            print "Finding solution to new problem:", key
            self.range = (self.minimumFaster(), self.solveFaster())
            print self.range
            self.answers[key] = self.range
            NK.answers.update(Experiments.loadConfiguration(constants["nkSolutions"]))
            Experiments.saveConfiguration(constants["nkSolutions"], NK.answers)
        '''
        random.seed(rngstate)
    def solve(self, direction):
        return direction(self.answers[self.key])

    def eval(self, child):
        child.fitness = 0
        for g, ep in enumerate(self.epistasis):
            child.fitness += self.fitness[g][int(''.join(map(str, [child.genes[x] for x in ep])), 2)]
        # self.updateAnswers(dict([(self.key, (child.fitness, child.fitness))]))
        # self.range = min(self.range[0], child.fitness), max(self.range[1], child.fitness)
        child.fitness = round((child.fitness - self.range[0]) / (self.range[1] - self.range[0]), 6)
    '''
    def __del__(self):
        try:
            existing = Experiments.loadConfiguration(self.solutions)
            if self.key not in existing or self.range[0] < existing[self.key][0] or existing[self.key][1] < self.range[1]:
                if self.key not in existing:
                    existing[self.key] = self.range
                else:
                    existing[self.key] = min(self.range[0], existing[self.key][0]), max(self.range[1], existing[self.key][1])
                Experiments.saveConfiguration(self.solutions, existing)
        except:
            print 'FAILED TO SAVE'
    '''
    def getFitness(self, g, genes):
        return self.fitness[g][int(''.join(map(str, genes)), 2)]

    def solveFaster(self):
        possible = [max(row) for row in self.fitness]
        def rowOptions(row, used, solution):
            changeable = [x for x in self.epistasis[row] if x not in used]
            options = []
            for count in range(2 ** len(changeable)):
                binary = bin(count)[2:]
                # Pad the number to be long enough
                binary = '0' * (len(changeable) - len(binary)) + binary
                testSolution = list(solution)
                for i, v in enumerate(changeable):
                    testSolution[v] = binary[i]
                fitness = self.fitness[row][int(''.join([testSolution[x] for x in self.epistasis[row]]), 2)]
                options.append((fitness, testSolution))
            best = max(options)[0]
            return (best, row, options)

        def recurse(usedRows, used, solution, currentFitness, best):
            if len(usedRows) >= self.n:
                return currentFitness, solution
            rows = []
            for i in range(self.n):
                if i not in usedRows:
                    rows.append(rowOptions(i, used, solution))

            workingRowsBest, index, options = max(rows)
            # workingRowsBest, index, options = min(rows, key=lambda X: len(X[2]))
            upperBound = sum([row[0] for row in rows]) - workingRowsBest
            changeable = [x for x in self.epistasis[index] if x not in used]
            bestFound = solution
            for fitness, testSolution in sorted(options, reverse=True):
                if currentFitness + fitness + upperBound > best:
                    quality, complete = recurse(usedRows + [index], used + changeable, testSolution, currentFitness + fitness, best)
                    if best < quality:
                        best = quality
                        bestFound = complete
                else:
                    break
            return best, bestFound
        quality, solution = recurse([], [], [''] * self.n, 0, 0)
        print quality, solution
        return quality

    def minimumFaster(self):
        possible = [min(row) for row in self.fitness]
        def rowOptions(row, used, solution):
            changeable = [x for x in self.epistasis[row] if x not in used]
            options = []
            for count in range(2 ** len(changeable)):
                binary = bin(count)[2:]
                # Pad the number to be long enough
                binary = '0' * (len(changeable) - len(binary)) + binary
                testSolution = list(solution)
                for i, v in enumerate(changeable):
                    testSolution[v] = binary[i]
                fitness = self.fitness[row][int(''.join([testSolution[x] for x in self.epistasis[row]]), 2)]
                options.append((fitness, testSolution))
            best = min(options)[0]
            return (best, row, options)

        def recurse(usedRows, used, solution, currentFitness, best):
            if len(usedRows) >= self.n:
                return currentFitness, solution
            rows = []
            for i in range(self.n):
                if i not in usedRows:
                    rows.append(rowOptions(i, used, solution))

            workingRowsBest, index, options = min(rows)
            upperBound = sum([row[0] for row in rows]) - workingRowsBest
            changeable = [x for x in self.epistasis[index] if x not in used]
            bestFound = solution
            for fitness, testSolution in sorted(options):
                if currentFitness + fitness + upperBound < best:
                    quality, complete = recurse(usedRows + [index], used + changeable, testSolution, currentFitness + fitness, best)
                    if quality < best:
                        best = quality
                        bestFound = complete
                else:
                    break
            return best, bestFound
        quality, solution = recurse([], [], [''] * self.n, 0, 1000)
        print quality, solution
        return quality

class NeighborNK(NK):
    def buildEpistasis(self):
        self.epistasis = [[(g + i) % self.n for i in range(self.k + 1)] for g in range(self.n)]
    def multiF(self, fitness, chunk, a, b):
        key = (chunk, a, b)
        try:
            return fitness[key]
        except:
            genewrap = (a + b) * 2
            fitness[key] = sum(self.getFitness(chunk * self.k + g, genewrap[g:g + self.k + 1]) for g in range(self.k))
            return fitness[key]

    def solve(self, direction):
        try:
            return direction(self.answers[self.key])
        except KeyError:
            print "Solvin"
            F = {}
            for n in range(self.n / self.k - 1, 1, -1):
                v, u = {}, {}
                for a in Util.binaryCounter(self.k):
                    for c in Util.binaryCounter(self.k):
                        u[a, c], v[a, c] = direction((self.multiF(F, n - 1, a, b) + self.multiF(F, n, b, c), b) for b in Util.binaryCounter(self.k))
                for a in Util.binaryCounter(self.k):
                    for c in Util.binaryCounter(self.k):
                        F[n - 1, a, c] = u[a, c]
                        F[n, a, c] = v[a, c]
            fitness, a, c = direction((self.multiF(F, 0, a, c) + self.multiF(F, 1, c, a), a, c) for c in Util.binaryCounter(self.k) for a in Util.binaryCounter(self.k))
            return fitness
            s = a + c
            last = c
            for i in range(2, self.n / self.k):
                last = F[i, last, a]
                s += last
            print s



class RiggedNK(NK):
    def __init__(self, constants):
        self.n = constants["dimensions"]
        self.k = constants["k"]
        rngstate = random.getstate()
        number = constants['problemSeed'] + constants["runNumber"]
        random.seed(number)
        indexes = range(self.n)
        self.fitness = [[random.random() for e in range(2 ** (self.k + 1))] for g in indexes]
        self.epistasis = [random.sample(indexes, self.k + 1) for g in indexes]
        for i, row in enumerate(self.epistasis):
            if i not in row:
                row[0] = i
        best = [random.randint(0, 1) for _ in range(self.n)]
        worst = [0 if x == 1 else 1 for x in best]
        self.range = (self.rig(worst, min), self.rig(best, max))
        random.seed(rngstate)

    def rig(self, genes, function):
        value = 0
        for g, ep in enumerate(self.epistasis):
            partial = self.fitness[g][int(''.join(map(str, [genes[x] for x in ep])), 2)] = function(self.fitness[g])
            value += partial
        return value

class Meta(FitnessFunction):
    def __init__(self, constants):
        self.constants = constants
        try:
            self.lowerConstants = Util.loadConfiguration(constants["lowerConfig"])
        except KeyError:
            self.lowerConstants = Util.loadConfigurations(sys.argv[2:])
            self.lowerConstants["name"] = self.lowerConstants["problemName"] + \
                                        '_' + self.lowerConstants["solverName"]

        constants["name"] = self.lowerConstants["name"]
        constants["maxFitness"] = (1.0, self.lowerConstants["maxFitness"], 0)
        self.mapping = dict((name, index) for index, name in enumerate(constants["mapping"]))
        self.evaled = {}
        self.bestEver = None
        self.tabled = 0

    def eval(self, child):
        self.scale(child, "parallelPopulations", 1, 50)
        self.scale(child, "popSize", 1, min(self.lowerConstants["evals"] / (2 * self.lowerConstants["parallelPopulations"]), 500))
        self.scale(child, "offSize", 1, min(self.lowerConstants["evals"] / (2 * self.lowerConstants["parallelPopulations"]), 500))
        self.scale(child, "parentTournament", 1, min(max(self.lowerConstants["popSize"] / 2, 1), 30))
        self.scale(child, "mutationRate", 0, 1, False)
        if self.lowerConstants["geneType"] == "FLT":
            self.scale(child, "mutationStepSize", 0, (self.lowerConstants["max"] - self.lowerConstants["min"]) / 4.0, False)
        if self.lowerConstants["crossover"] == "npoint":
            self.scale(child, "numberOfPoints", 1, self.lowerConstants["dimensions"] - 1)
        if self.lowerConstants["crossover"] == "scx":
            self.scale(child, "initialCrossoverLength", 1, self.lowerConstants["dimensions"] * 2)
        if self.lowerConstants["popType"] == "SuCo" and self.lowerConstants["SuCoLevel"] == "Support":
            self.scale(child, "support_popSize", 1, min(self.lowerConstants["evals"] / 2, 500))
            self.scale(child, "support_offSize", 1, min(self.lowerConstants["evals"] / 2, 500))
            self.scale(child, "support_parentTournament", 1, min(max(self.lowerConstants["popSize"] / 2, 1), 30))
            if 'repeat' in self.lowerConstants["SuCoFitnessDecorators"]:
                self.scale(child, "evalsPerGeneration", 1, 50)
        for key in self.constants["mapping"]:
            if key not in self.lowerConstants:
                self.lowerConstants[key] = 0

        hashable = "\n".join('"%s":%s,' % (key, str(self.lowerConstants[key])) for key in self.constants["mapping"])
        try:
            child.fitness = self.evaled[hashable]
            self.tabled += 1
            print "TABELED"
        except KeyError:
            child.fitness = Experiments.basic(self.lowerConstants)
            self.evaled[hashable] = child.fitness

        print 'Meta Evals:', len(self.evaled)
        if self.bestEver is None or self.bestEver < child:
            self.bestEver = child
            # sr, mf, mes, avr, std = child.fitness
            # filename = "../Meta/%f_%f_%f_%f_%f_%i_%s_%s%s" % (sr, mf, mes, avr, std, len(self.evaled), self.lowerConstants["problem"], self.constants["name"], support)
            mean, std, mes = child.fitness
            data = {"mean":abs(mean), "std":std, "metaEvals":len(self.evaled), "problem":self.lowerConstants['problem'], "name":self.constants['name'],
                    "evals":mes}
            filename = "../Meta/%(mean)f_%(std)f_%(evals)i_%(metaEvals)i_%(problem)s_%(name)s" % data
            Util.saveConfiguration(filename, self.lowerConstants)

    def scale(self, child, name, lower, upper, makeInt=True):
        if name in self.mapping:
            self.lowerConstants[name] = (upper - lower) * child.genes[self.mapping[name]] + lower
            if makeInt: self.lowerConstants[name] = int(self.lowerConstants[name])
