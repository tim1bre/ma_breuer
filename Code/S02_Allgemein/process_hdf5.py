"""
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def hdf5_to_df(path, filename):
    convert metrices and attributes from hdf5 to df

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
                    if col == 'dist_name':
                        df.loc[i_exp, col+'1'] = hdf['experiments/'+exp].attrs[col][0]
                        df.loc[i_exp, col+'2'] = hdf['experiments/'+exp].attrs[col][1]
                    else:
                        df.loc[i_exp, col] = hdf['experiments/'+exp].attrs[col]

            # get signal power of disturbance
            dist_name = hdf['experiments/'+exp].attrs['dist_name']
            df.loc[i_exp, 'signal_power1'] = hdf['disturbances/dist1/'+str(dist_name[0])].attrs['signalpower']
            df.loc[i_exp, 'signal_power2'] = hdf['disturbances/dist1/'+str(dist_name[1])].attrs['signalpower']

            # check if experiment needs to be "disqualified"
            # 0.1s before disturbance there shouldn't be a false positive
            if (hdf['experiments/'+exp+'/data/pdist0'][()][0][9989] <= 0) and (hdf['experiments/'+exp+'/data/y0_filt'][()][0][9989] <= 20.1) and (hdf['experiments/'+exp+'/data/pdist0'][()][0][19989] <= 0) and (hdf['experiments/'+exp+'/data/y0_filt'][()][0][19989] <= 20.1):
                df.loc[i_exp, 'disq'] = 0
            else:
                df.loc[i_exp, 'disq'] = 1
        
        new_filename = filename.split('.')[0]+'Frame.pkl'
        df.to_pickle(path + new_filename)

def plot_from_exp_id(path, filename, exp_id):
    """ create subplot with most important data """

    with h5py.File(path+filename, 'r') as hdf:
    
        dist1 = hdf['experiments/exp' + str(exp_id)].attrs['dist_name'][0]
        dist2 = hdf['experiments/exp' + str(exp_id)].attrs['dist_name'][1]
        te = hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0][-1]
        t_dist = [0] + list(np.transpose(hdf['disturbances/dist1/'+dist1][()])[0]) + list(np.transpose(hdf['disturbances/dist2/'+dist2][()])[0]) + [te]
        valve_dist = [1] + list(np.transpose(hdf['disturbances/dist1/'+dist1][()])[1]) + list(np.transpose(hdf['disturbances/dist2/'+dist2][()])[1]) + [1]

        # create subplot
        fig1, axs1 = plt.subplots(5, 1, sharex=True)
        fig1.suptitle('noise: ' + str(hdf['experiments/exp'+str(exp_id)].attrs['noise_sigma']) + ', deadtime: ' + str(hdf['experiments/exp'+str(exp_id)].attrs['deadtime']) + ', mode: ' + hdf['experiments/exp'+str(exp_id)].attrs['detect_mode'])

        axs1[0].set(ylabel='Ventil')
        axs1[0].plot(t_dist, valve_dist, 'r-')
        axs1[1].set(ylabel='Füllstand')
        axs1[1].plot(hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0], hdf['experiments/exp' + str(exp_id) +'/data/y0_filt'][()][0], 'r-')
        axs1[1].set_ylim(19, 21.5)
        axs1[1].plot( [hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0][0], hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0][-1]], [20, 20], color="black")
        axs1[2].set(ylabel='Pdist')
        axs1[2].plot(hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0], hdf['experiments/exp' + str(exp_id) +'/data/pdist0'][()][0], 'r-')
        axs1[3].set(ylabel='Sollwert')
        axs1[3].plot(hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0], hdf['experiments/exp' + str(exp_id) +'/data/pi0'][()][0], 'r-')
        axs1[4].set(ylabel='Stellgröße')
        axs1[4].plot(hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0], hdf['experiments/exp' + str(exp_id) +'/data/u0'][()][0], 'r-')

        print("Plot mit gefiltertem Füllstand!")

        plt.show()

def hdf5_to_df_MC3(path, filename):
    """ convert metrices and attributes from hdf5 to df """

    with h5py.File(path+filename, 'r') as hdf:

        attributes = list(hdf['experiments/exp1'].attrs.keys())
        metrices = ['overlap', 'resloss_sum', 'rt_noise1'] #, 'time_to_detect', 'time_to_recover']

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
                            df.loc[i_exp, col+str(i+1)] = hdf['experiments/'+exp+'/metrices/'+col][()]
                    else:
                        df.loc[i_exp, col] = hdf['experiments/'+exp+'/metrices/'+col][()]

                # fill in attributes
                elif col in attributes:
                    if col == 'dist_name':
                        df.loc[i_exp, col+'1'] = hdf['experiments/'+exp].attrs[col]
                        #df.loc[i_exp, col+'2'] = hdf['experiments/'+exp].attrs[col][1]
                    else:
                        df.loc[i_exp, col] = hdf['experiments/'+exp].attrs[col]

            # get signal power of disturbance
            #dist_name = hdf['experiments/'+exp].attrs['dist_name']
            #df.loc[i_exp, 'signal_power1'] = hdf['disturbances/dist1/'+str(dist_name[0])].attrs['signalpower']
            #df.loc[i_exp, 'signal_power2'] = hdf['disturbances/dist1/'+str(dist_name[1])].attrs['signalpower']

            # check if experiment needs to be "disqualified"
            # 0.1s before disturbance there shouldn't be a false positive
            if (hdf['experiments/'+exp+'/data/pdist0'][()][0][9989] <= 0) and (hdf['experiments/'+exp+'/data/y0_filt'][()][0][9989] <= 20.1) and (hdf['experiments/'+exp+'/data/pdist0'][()][0][19989] <= 0) and (hdf['experiments/'+exp+'/data/y0_filt'][()][0][19989] <= 20.1):
                df.loc[i_exp, 'disq'] = 0
            else:
                df.loc[i_exp, 'disq'] = 1
        
        new_filename = filename.split('.')[0]+'Frame.pkl'
        df.to_pickle(path + new_filename)

def plot_from_exp_id_onedist(path, filename, exp_id):
    """ create subplot with most important data """

    with h5py.File(path+filename, 'r') as hdf:
    
        dist1 = hdf['experiments/exp' + str(exp_id)].attrs['dist_name']
        #dist2 = hdf['experiments/exp' + str(exp_id)].attrs['dist_name'][1]
        te = hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0][-1]
        t_dist = [0] + list(np.transpose(hdf['disturbances/dist1/'+dist1][()])[0]) + [te]
        valve_dist = [1] + list(np.transpose(hdf['disturbances/dist1/'+dist1][()])[1]) + [1]

        # create subplot
        fig1, axs1 = plt.subplots(5, 1, sharex=True)
        fig1.suptitle('noise: ' + str(hdf['experiments/exp'+str(exp_id)].attrs['noise_sigma']) + ', deadtime: ' + str(hdf['experiments/exp'+str(exp_id)].attrs['deadtime']) + ', mode: ' + hdf['experiments/exp'+str(exp_id)].attrs['detect_mode'])

        axs1[0].set(ylabel='Ventil')
        axs1[0].plot(t_dist, valve_dist, 'r-')
        axs1[1].set(ylabel='Füllstand')
        axs1[1].plot(hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0], hdf['experiments/exp' + str(exp_id) +'/data/y0_filt'][()][0], 'r-')
        axs1[1].set_ylim(19, 21.5)
        axs1[1].plot( [hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0][0], hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0][-1]], [20, 20], color="black")
        axs1[2].set(ylabel='Pdist')
        axs1[2].plot(hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0], hdf['experiments/exp' + str(exp_id) +'/data/pdist0'][()][0], 'r-')
        axs1[3].set(ylabel='Sollwert')
        axs1[3].plot(hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0], hdf['experiments/exp' + str(exp_id) +'/data/pi0'][()][0], 'r-')
        axs1[4].set(ylabel='Stellgröße')
        axs1[4].plot(hdf['experiments/exp' + str(exp_id) +'/data/t'][()][0], hdf['experiments/exp' + str(exp_id) +'/data/u0'][()][0], 'r-')

        print("Plot mit gefiltertem Füllstand!")

        plt.show()

def main():

    mypath = r"C:\Users\timbr\OneDrive\Masterarbeit\Daten\2020_09_13_Monte_Carlo_3\\"
    filename = "2020_09_14__14_14_46_Monte_Carlo_Data.hdf5"

    hdf5_to_df_MC3(mypath, filename)
    #plot_from_exp_id(mypath, filename, 1)

if __name__ == '__main__':
    main()

"""