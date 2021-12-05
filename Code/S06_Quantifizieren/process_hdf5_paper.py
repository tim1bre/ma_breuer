import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def hdf5_to_df(filename, path=""):
    """ convert metrices and attributes from hdf5 to df """

    with h5py.File(path+filename, 'r') as hdf:

        attributes = list(hdf['experiments/exp1'].attrs.keys())
        metrices = list(hdf['experiments/exp1/metrices'].keys())

        # init df with NaN
        df = pd.DataFrame(columns=attributes+metrices+['signal_power'], index=list(range(0, len(hdf['experiments']))))

        # fill df with metrices and attributes
        for i_exp, exp in enumerate(hdf['experiments'].keys()):

            for i_col, col in enumerate(df.columns):
                
                # fill in metrices
                if col in metrices:
                    # avoid arrays around lists
                    if hasattr(hdf['experiments/'+exp+'/metrices/'+col][()], '__len__') and (not isinstance(hdf['experiments/'+exp+'/metrices/'+col][()], str)):

                        for i in range(0,2):
                            df.loc[i_exp, col+str(i+1)] = hdf['experiments/'+exp+'/metrices/'+col][()][0][i]
                    else:
                        df.loc[i_exp, col] = hdf['experiments/'+exp+'/metrices/'+col][()]

                # fill in attributes
                elif col in attributes:
                    df.loc[i_exp, col] = hdf['experiments/'+exp].attrs[col]

            # get signal power of disturbance
            dist_name = hdf['experiments/'+exp].attrs['dist_name']
            df.loc[i_exp, 'signal_power'] = hdf['disturbances/dist1/'+str(dist_name)].attrs['signalpower']

        new_filename = filename.split('.')[0]+'.csv'
        df.to_csv(path + new_filename, sep=';')

def plot_from_exp_id(path, filename, exp_id, ymin=0.16, ymax=0.25, t0=0, te=900):
    """ create subplot with most important data """

    with h5py.File(path+filename, 'r') as hdf:
    
        dist1 = hdf['experiments/exp' + str(exp_id)].attrs['dist_name']
        te = hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0][-1]
        t_dist = [0] + list(np.transpose(hdf['disturbances/dist1/'+dist1][()])[0]) + [te]
        valve_dist = [1] + list(np.transpose(hdf['disturbances/dist1/'+dist1][()])[1]) + [1]

        # create subplot
        fig1, axs1 = plt.subplots(5, 1, sharex=True)
        fig1.suptitle('noise: ' + str(hdf['experiments/exp'+str(exp_id)].attrs['noise_sigma']) + ', deadtime: ' + str(hdf['experiments/exp'+str(exp_id)].attrs['deadtime']) + ', mode: ' + hdf['experiments/exp'+str(exp_id)].attrs['detect_mode'])

        axs1[0].set(ylabel='Ventil')
        axs1[0].plot(t_dist, valve_dist, 'r-')
        axs1[0].set_xlim(t0, te)
        axs1[1].set(ylabel='Füllstand')
        axs1[1].plot(hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0], hdf['experiments/exp' + str(exp_id) +'/data/y0_filt'][()][0], 'r-')
        axs1[1].set_ylim(ymin, ymax)
        axs1[1].plot( [hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0][0], hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0][-1]], [0.2, 0.2], color="black")
        axs1[2].set(ylabel='Pdist')
        axs1[2].plot(hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0], hdf['experiments/exp' + str(exp_id) +'/data/pdist0'][()][0], 'r-')
        axs1[3].set(ylabel='Sollwert')
        axs1[3].plot(hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0], hdf['experiments/exp' + str(exp_id) +'/data/pi0'][()][0], 'r-')
        axs1[4].set(ylabel='Stellgröße')
        axs1[4].plot(hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0], hdf['experiments/exp' + str(exp_id) +'/data/u0'][()][0], 'r-')

        plt.show()


def main():
    hdf5_to_df("2020_12_22__15_47_14_Monte_Carlo_Data.hdf5")

if __name__ == '__main__':
    main()