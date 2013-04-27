"""
This module contains algorithms for data analysis
"""
import math

def median(data, default=0):
	"""
	Computes the median
	
	Parameters:
	
	-``data``: The data or results from an experiment.
	
	-``default``: The default value to return if the size of the data is 0
	"""
	ordered = sorted(data)
	size = len(ordered)
	if size == 0:
		return default
	elif size % 2 == 1 :
		return ordered[(size - 1) / 2]
	else:
		return (ordered[(size / 2)] + ordered[size / 2 - 1 ]) / 2.0

def meanstd(data):
	"""
	Computes the mean and standard deviation

	Parameters:

	-``data``: The data or results from an experiment.
	"""
	mean = float(sum(data)) / len(data)
	std = math.sqrt(sum([(value - mean) ** 2 for value in data]) / len(data))
	return mean, std
