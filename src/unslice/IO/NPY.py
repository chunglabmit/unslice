import numpy

import clarity.IO as io

def writePoints(filename, points, **args):
    numpy.save(filename, points)
    return filename


def readPoints(filename, **args):
    points = numpy.load(filename)
    return io.pointsToRange(points, **args)

