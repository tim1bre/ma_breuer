import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from Code.S02_Allgemein.process_hdf5 import plot_from_exp_id

def spiderplot_restriang(path, filename):
    """ create spiderplot for paper """

    #h5_filename = filename+".hdf5"
    csv_filename = filename + ".csv"

    # read data
    df_raw = pd.read_csv(path+csv_filename, sep=';')

    # convert columns to numerical values
    numeric_columns = ['exp_id', 'res_triang', 'res_triang_noise', 'max_loss', 'ttd', 'mttr', 'fn', 'fp',  'signal_power', 'sim_time', 'dist_id', 'disq']
    for i in numeric_columns:
        df_raw[i] = df_raw[i].apply(pd.to_numeric, errors='coerce')

    # select columns of DataFrame that are used during the analysis
    df = df_raw[['exp_id', 'detect_mode', 'dist_name', 'res_triang', 'res_triang_noise', 'max_loss', 'ttd', 'mttr', 'fn', 'fp', 'disq', 'signal_power','sim_time', 'dist_id']]
    df['ttd'].fillna(400, inplace=True)

    dist_types = []
    for i in df['dist_name'].str.split("_"):
        if i[0] not in dist_types:
            dist_types.append(i[0])

    detect_modes = list(df.detect_mode.unique())
    detect_modes.remove('detect_dummy')

    df_dummy = df[df['detect_mode'] == 'detect_dummy']
    dummy_ref= df_dummy['res_triang'].mean()

    data_set = []
    for mode in detect_modes:
        df_select = df[df['detect_mode']==mode]
        new_line = []
        for j in dist_types:
            df_select2 = df_select[df_select['dist_name'].str.contains(j)]
            restriang = df_select2['res_triang'].mean()

            new_line.append(restriang/ dummy_ref)
        data_set.append(new_line)
    
    labels = np.array(dist_types)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    
    stats = data_set
    angles = np.concatenate((angles,[angles[0]]))
    name = "Spiderplot Resilience Triangle Normalized"
    markers = [0.5, 1]
    str_markers = ["0", "1", "2", "3", "4"]

    fig= plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, polar=True)
    
    # plot data
    for data in stats:
        data.append(data[0]) 
    
    ax.plot(angles, stats[0], 'x-.', linewidth=1, color='black', markersize=4)
    ax.plot(angles, stats[1], 'x:', linewidth=1, color='dimgray', markersize=4)
    ax.plot(angles, stats[2], 'x-', linewidth=1, color='black', markersize=4)
    ax.plot(angles, stats[3], 'x-', linewidth=1, color='gray', markersize=4)
    ax.plot(angles, stats[4], 'x--', linewidth=1, color='gray', markersize=4)

    # overwrite labels for plot
    labels[np.where(labels == 'triag')] = 'triangle'
    labels[np.where(labels == 'varamp')] = 'variable amplitude'

    ax.set_thetagrids(angles * 180/np.pi, labels)
    plt.yticks(markers)
    ax.set_title(name)
    ax.grid(True)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(11)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(11)

    plt.yticks(fontname = "Verdana")
    plt.xticks(fontname = "Verdana")

    # plt.legend(detect_modes, loc='right', bbox_to_anchor=(1.5, 0.9))
    plt.legend(['modelbased', 'DTW', 'gradientbased', 'PELT', 'ARIMA'], loc='right', bbox_to_anchor=(1.5, 0.9))

    return plt.show()

def main():
    path = r"C:\Users\timbr\OneDrive\Masterarbeit\Daten\Paper\\"
    filename = "2020_12_29__18_11_36_Monte_Carlo_Data"
    spiderplot_restriang(path, filename)

if __name__ == '__main__':
    main()
    
    