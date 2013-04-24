import inspect
import json
import itertools

def childClasses(objType):
    children = []
    for x in objType.__subclasses__():
        children.extend(childClasses(x) + [x])
    return children

def moduleFunctions(module):
    return dict(inspect.getmembers(module, inspect.isroutine))

def moduleClasses(module):
    return dict(inspect.getmembers(module, inspect.isclass))

def classMethods(classType):
    return dict(inspect.getmembers(classType, inspect.ismethod))

def flip(number):
    return 1 - number

def loadConfiguration(filename):
    with open(filename) as f:
        return json.load(f)

def loadConfigurations(filenames):
    result = {}
    for filename in filenames:
        result.update(loadConfiguration(filename))
    return result

def saveConfiguration(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

def binaryCounter(bits):
    return itertools.product((0, 1), repeat=bits)
