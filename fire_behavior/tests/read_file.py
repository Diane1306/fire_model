import numpy as np  
import os
import scipy.io as si

def read_file(filename):
    '''
        Read input file with given name.

        Args:
            filename (string): full path for input file
        Returns:
            (numpy array): the content of the file
    '''
    if not os.path.isfile(filename):
        raise ValueError("Input file does not exist: {0}. I'll quit now.".format(filename))

    # code to load and parse the data from input file
    data = si.loadmat(filename)

    if not (len(data.keys())-3):
        # the data should not be empty then the dict should be have more than 3 keys
        raise ValueError("No data in input file.")

    return data