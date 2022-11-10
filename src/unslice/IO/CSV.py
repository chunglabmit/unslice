import numpy

import hubris.IO as io


def writePoints(filename, points, **args):
    """Write point data to csv file
    
    Arguments:
        filename (str): file name
        points (array): point data
    
    Returns:
        str: file name
    """
    
    numpy.savetxt(filename, points, delimiter=',', newline='\n', fmt='%.5e')
    return filename


def readPoints(filename, **args):
    """Read point data to csv file
    
    Arguments:
        filename (str): file name
        **args: arguments for :func:`~clarity.IO.pointsToRange`
    
    Returns:
        str: file name
    """
    
    points = numpy.loadtxt(filename, delimiter=',');
    return io.pointsToRange(points, **args);
