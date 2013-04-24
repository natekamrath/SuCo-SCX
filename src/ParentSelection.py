import random
def tournament(individuals, number, constants):
    return [max(random.sample(individuals, min(len(individuals), constants["parentTournament"])))
            for _ in range(number)]

def elitist(individuals, number, constants={}):
    return sorted(individuals, reverse=True)[:number]
