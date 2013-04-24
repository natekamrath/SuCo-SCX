import math
def combine(evalsToFitness, constants):
    levels = sorted(set([level[1] for run in evalsToFitness for level in run]))
    evals = [0] * len(levels)
    successes = [0] * len(levels)
    for i, level in enumerate(levels):
        for run in evalsToFitness:
            for step in run:
                if level <= step[1]:
                    successes[i] += 1
                    evals[i] += step[0]
                    break
    for i, count in enumerate(successes):
        evals[i] /= float(count)
        successes[i] = float(count) / len(evalsToFitness)
    combined = zip(levels, evals, successes)
    return combined[:-1:max(1, len(levels) / constants["logLines"])] + [combined[-1]]

def successRate(evalsToFitness, constants):
    return sum([1 for _, fitness in evalsToFitness if fitness >= constants["maxFitness"]]) / float(len(evalsToFitness))

def median(data, default=0):
    ordered = sorted(data)
    size = len(ordered)
    if size == 0:
        return default
    elif size % 2 == 1 :
        return ordered[(size - 1) / 2]
    else:
        return (ordered[(size / 2)] + ordered[size / 2 - 1 ]) / 2.0

def medianUnsuccessful(evalsToFitness, constants):
    return median([fitness for _, fitness in evalsToFitness if fitness < constants["maxFitness"]], constants["maxFitness"])

def medianEvalsToSuccess(evalsToFitness, constants):
    # Get the best fitness achieved
    successful = [evals for evals, fitness in evalsToFitness if fitness >= constants["maxFitness"]]
    if len(successful) == 0:
        middle = medianUnsuccessful(evalsToFitness, constants)
        successful = [evals for evals, fitness in evalsToFitness if fitness >= middle]
    return median(successful)

def meanstd(data):
    mean = float(sum(data)) / len(data)
    std = math.sqrt(sum([(value - mean) ** 2 for value in data]) / len(data))
    return mean, std

def ltgafilter(results, constants):
    successes = [run for run in results if run['fitness'] >= constants["maxFitness"]]
    rate = float(len(successes)) / constants["runs"]
    mintrap = meanstd([run["mintrap"] for run in successes])
    evals = meanstd([run["evals"] for run in successes])
    local = meanstd([run["local"] for run in successes])
    total = meanstd([run["total"] for run in successes])
    return rate, mintrap, local, evals, total

def metaFilter(evalsToSuccess, constants):
    top = [run[-1][0:2] for run in evalsToSuccess]
    try:
        mean = sum([fit for eval, fit in top]) / len(top)
        std = math.sqrt(sum([(fit - mean) ** 2 for eval, fit in top]) / len(top))
    except TypeError:
        mean = successRate(top, constants)
        std = medianUnsuccessful(top, constants)
    #return (mean, -std, -medianEvalsToSuccess(top, constants))
    return (successRate(top, constants), medianUnsuccessful(top, constants), -medianEvalsToSuccess(top, constants), mean, std)
