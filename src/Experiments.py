"""
This module contains code for running experiments
"""

import os
import Population
import Util
import Fitness
import random
import Analysis
import GeneTypes
from Population import Individual
import Crossover

def oneRun(constants, evaluation, sheet = None):
	"""
	Performs one run of the experiment
	
	Parameters:
	
	-``constants``: Config settings for the EA from the config files specified at run time
	
	-``evaluation``: Fitness function.  From ``Fitness`` Module
	"""
	created = []
	if "popType" in constants:
		pop = Util.moduleClasses(Population)[constants["popType"]](constants)
	else:
		pop = Population.Population(constants)

	best, evals, lastImproved = Population.Individual(), 0, 0
	rowCounter = 0
	while evals < constants["evals"] and best.fitness < constants["maxFitness"]:
		try:
			child = pop.next()
		except StopIteration:
			break
		evaluation(child)
		evals += 1
		if best < child:
			lastImproved = evals
			created.append((evals, child.fitness, child))
			best = child
	print best.fitness
	return created

def basic(constants):
	"""
	Runs a basic experiment
	
	Parameters:
	
	-``constants``: Config settings for the EA from the config files specified at run time
	"""
	try: random.seed(constants["seed"])
	except KeyError: pass

	results = []

	for run in range(constants["runs"]):
		constants["runNumber"] = run
		evaluation = Util.moduleClasses(Fitness)[constants["problem"]](constants).eval
		results.append(oneRun(constants, evaluation))
	data = []
	for i in results:
		data.append(i[-1][1])
	return Analysis.meanstd(data), results
	


if __name__ == '__main__':
	import sys
	config = Util.loadConfigurations(sys.argv[1:])
	filtered, raw = basic(config)
	print config["name"]
	with open("../LogFiles/" + config["name"] + ".csv", 'w') as f:
		f.write(str(filtered) + "\n")
		for run in raw:
			f.write(str(run) + "\n")
		print config["name"], "results: ", str(filtered)

   
