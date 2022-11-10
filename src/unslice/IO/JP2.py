import glymur 
import numpy
import hubris.IO as io
import glob, os 

def read_jp2(filepath, x = all, y = all, num_threads=24):
    jp2 = glymur.Jp2k(filepath)
    glymur.set_option('lib.num_threads',num_threads)
    if x is all and y is all:
        fullres = jp2[:]
    elif y is all: 
        fullres = jp2[x[0]:x[1]]
    elif x is all:
        fullres = jp2[:,y[0]:y[1]]
    else:
        fullres = jp2[x[0]:x[1],y[0]:y[1]] 

    return fullres 

def convert_filenames_regex(filepath):
    # Find curly braces and convert it to something readable by glob.glob
    idx = filepath.find("{")
    num_repeats = int(filepath[idx+1])-1
    str_to_add = filepath[idx-5:idx]*num_repeats
    return filepath[:idx]+str_to_add+filepath[idx+3:]

def readData(filename, x = all, y = all, z = all, **args):
    """Read data from a single tif image or stack
    
    Arguments:
        filename (str): file name as regular expression
        x,y,z (tuple): data range specifications
    
    Returns:
        array: image data
    """

    # if it's a 2D image 
    if os.path.isfile(filename):
        final_img = read_jp2(filename, x=y, y=x) # needs to be transposed 
        final_img = final_img.transpose([1,0])
        return final_img 
    else:
        if "{" in filename:
            filename = convert_filenames_regex(filename)
        filenames_list = sorted(glob.glob(filename)) 

        dataSize = read_jp2(filenames_list[0]).shape +(len(filenames_list),)
        data = np.zeros(dataSize[:2]+(0,),dtype='uint16')
        if z is all:
            filenames_list_ = filenames_list 
        else: 
            filenames_list_ = filenames_list[z[0]:z[1]]
            
        for file in filenames_list_:
            new_img = read_jp2(file,x=y,y=x)
            data = np.concatenate((data,new_img),axis=2)
        data = data.transpose([1,0,2])
        return data 
