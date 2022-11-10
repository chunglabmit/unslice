import numpy
import tifffile as tiff
import warnings 
warnings.filterwarnings('ignore')
import hubris.IO as io


def dataSize(filename, **args):
    """Returns size of data in tif file
    
    Arguments:
        filename (str): file name as regular expression
        x,y,z (tuple): data range specifications
    
    Returns:
        tuple: data size
    """

    t = tiff.TiffFile(filename)# fastij=False)
    d3 = len(t.pages)
    d2 = t.pages[0].shape
    d2 = (d2[1], d2[0])
    if d3 > 1:
        dims = d2 + (d3,)
    else:
        dims =  d2
    
    return io.dataSizeFromDataRange(dims, **args)

def dataZSize(filename, z = all, **args):
    """Returns z size of data in tif file
    
    Arguments:
        filename (str): file name as regular expression
        z (tuple): z data range specification
    
    Returns:
        int: z data size
    """
    
    t = tiff.TiffFile(filename)#, fastij=False)
    d3 = len(t.pages)
    if d3 > 1:
        return io.toDataSize(d3, r = z)
    else:
        return None



def readData(filename, x = all, y = all, z = all, **args):
    """Read data from a single tif image or stack
    
    Arguments:
        filename (str): file name as regular expression
        x,y,z (tuple): data range specifications
    
    Returns:
        array: image data
    """
    
    dsize = dataSize(filename)
    
    if len(dsize) == 2:
        data = tiff.imread(filename, key = 0)
        return io.dataToRange(data.transpose([1,0]), x = x, y = y)
        
    else:
        if z is all:
            data = tiff.imread(filename)
            if data.ndim == 2:
                data = data.transpose([1,0])
            elif data.ndim == 3:
                data = data.transpose([2,1,0])
            elif data.ndim == 4:
                data = data.transpose([2,1,0,3])
            else:
                raise RuntimeError('readData: dimension %d not supproted!' % data.ndim)
            return io.dataToRange(data, x = x, y = y, z = all)
        
        else: #optimize for z ranges
            ds = io.dataSizeFromDataRange(dsize, x = x, y = y, z = z)
            t = tiff.TiffFile(filename)
            p = t.pages[0]
            data = numpy.zeros(ds, dtype = p.dtype)
            rz = io.toDataRange(dsize[2], r = z)
            
            for i in range(rz[0], rz[1]):
                ## WG Edit: 11/3/2017 - extremely useful for debugging. 
                # print(t.pages[0])
                # print(rz[0], rz[1])
                # print(i)
                # print(len(t.pages))
                xydata = t.pages[i].asarray()
                data[:,:,i-rz[0]] = io.dataToRange(xydata.transpose([1,0]), x = x, y = y)
            
            return data


def writeData(filename, data):
    """Write image data to tif file
    
    Arguments:
        filename (str): file name 
        data (array): image data
    
    Returns:
        str: tif file name
    """
    
    d = len(data.shape)
    
    if d == 2:
        tiff.imsave(filename, data.transpose([1,0]))
    elif d == 3:
        tiff.imsave(filename, data.transpose([2,1,0]))
    elif d == 4:
        t = tiff.TiffWriter(filename, bigtiff = True)
        t.save(data.transpose([2,1,0,3]), photometric = 'minisblack',  planarconfig = 'contig')
        t.close()
    else:
        raise RuntimeError('writing multiple channel data to tif not supported')
    
    return filename
    

def copyData(source, sink):
    """Copy a data file from source to sink
    
    Arguments:
        source (str): file name pattern of source
        sink (str): file name pattern of sink
    
    Returns:
        str: file name of the copy
    """ 
    
    io.copyFile(source, sink)
