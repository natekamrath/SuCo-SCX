import random
def tournament(individuals, number, constants):
    while len(individuals) > number:
        # Get the indexes of the tournament competitors
        tourn = random.sample(xrange(len(individuals)), min(len(individuals), constants["survivorTournament"]))
        # find the minimum, pairing individual and index
        toRemove = min((individuals[i], i) for i in tourn)
        # Swap the last guy up to the worst guy's position
        individuals[toRemove[1]], individuals[-1] = individuals[-1], None
        # shorten the list
        del individuals[-1]
    return individuals

def elitist(individuals, number, constants={}):
    return sorted(individuals)[-number:]

def ageist(individuals, number, constants={}):
    return sorted(individuals, key=lambda x: x.id)[-number:]
