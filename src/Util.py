"""
This module contains utilities and tools for gathering information about classes, class methods, and functions.
"""
import inspect
import json
import itertools

def childClasses(objType):
	"""
	Returns all the child classes of the specified class type.

	Parameters:

	-``objType``: They object type whose children are to be found.
	"""
	children = []
	for x in objType.__subclasses__():
		children.extend(childClasses(x) + [x])
	return children

def moduleFunctions(module):
	"""
	Returns a dictionary of functions from the specified module.

	Parameters:

	-``module``: The module to get functions from. 
	"""
	return dict(inspect.getmembers(module, inspect.isroutine))

def moduleClasses(module):
	"""
	Returns a dictionary of classes from the specified module.

	Parameters:

	-``module``: The module to get classes from.
	"""
	return dict(inspect.getmembers(module, inspect.isclass))

def classMethods(classType):
	"""
	Returns a dictionary of methods within a certain class.

	Parameters:

	-``classType``: The class whose methods are being retrieved.
	"""
	return dict(inspect.getmembers(classType, inspect.ismethod))

def flip(number):
	"""
	Flips a number

	Parameters:

	-``number``: The number to be flipped.
	"""
	return 1 - number

def loadConfiguration(filename):
	"""
	Loads a configuration file.

	Parameters:

	-``filename``: the file name of the configuration file to be loaded.
	"""
	with open(filename) as f:
		return json.load(f)

def loadConfigurations(filenames):
	"""
	Loads multiple configuration files.

	Parameters:

	-``filenames``:  list of file names for the configuration files to be loaded.
	"""
	result = {}
	for filename in filenames:
		result.update(loadConfiguration(filename))
	return result

def saveConfiguration(filename, data):
	"""
	Saves a configuration file

	Parameters:

	-``filename``: The name of the file to save the configs to.

	-``data``: The data to write to the config file being saved.
	"""
	with open(filename, 'w') as f:
		json.dump(data, f)
