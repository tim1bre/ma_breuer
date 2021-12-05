from Code.S02_Allgemein.save_object import load_object
from Code.S06_Quantifizieren.evaluate_all_metrices import evaluate_all, evaluate_hdf5
from os import listdir
from os.path import isfile, join
import h5py

h5_filename = 'test_debug.hdf5'

mypath = r"C:\Users\timbr\OneDrive\Masterarbeit\Daten\2020_07_21_Monte Carlo Testlauf v5\\"

f = listdir(mypath)
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

obj_list = []

df = None
df_list = None

for i in onlyfiles:

    tanksys = load_object(mypath + '\\' + str(i))
    tanksys = evaluate_all(tanksys, 95)
    [a, b, c] = evaluate_hdf5(tanksys)

with h5py.File(h5_filename, 'w') as hdf:

    G_exp = hdf.create_group("experiments")
    
    print('enter loop') 

    [attributes, data_h5, metrices] = [a, b, c]
    m = [attributes, data_h5, metrices]

    if isinstance(m, str):
        print('str')
    else:
        print('no string') 
        
    # add new experiment
    exp_id = len(hdf['experiments'])+1
    new_exp = G_exp.create_group("exp"+str(exp_id))

    new_exp_data = new_exp.create_group("data")
    for i in data_h5.keys():
        new_exp_data.create_dataset(i, shape=(1,len(data_h5[i])), data=data_h5[i])

    new_exp_metrices = new_exp.create_group("metrices")
    for i in metrices.keys():
        if (isinstance(metrices[i], list)):
            new_exp_metrices.create_dataset(i, shape=(1, len(metrices[i])), data=metrices[i])
        else:
            new_exp_metrices.create_dataset(i, data=metrices[i])

    

    for i in attributes.keys():
        new_exp.attrs[i] = attributes[i]
    new_exp.attrs['exp_id'] = exp_id