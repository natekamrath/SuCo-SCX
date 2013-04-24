import random
class GeneType(object):
    def __init__(self, minimum, maximum):
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
    def randomValue(self): return random.randint(self.min, self.max)
    def fixGene(self, gene):
        value = int(gene)
        if value != gene and random.random() < gene - value:
            value += 1
        return value

class FLT(GeneType):
    def randomValue(self): return random.uniform(self.min, self.max)

