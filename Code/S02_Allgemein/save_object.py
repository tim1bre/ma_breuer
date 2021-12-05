import os
from datetime import datetime
import dill

def save_object(filename, object):
    """ save class object as .pkl """

    path = r"C:\Users\timbr\OneDrive\Masterarbeit\Daten\\"
    filename = path + filename + datetime.now().strftime('%Y_%m_%d__%H_%M_%S__%fZ')[:-3]+'.pkl'
    
    with open(filename, 'wb') as f:
        dill.dump(object, f)

def load_object(filename):
    """ load class object as pkl """
	
    input_file = open(filename, 'rb')
    object = dill.load(input_file)

    return object 