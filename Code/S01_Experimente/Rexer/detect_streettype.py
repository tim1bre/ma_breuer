import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import ruptures as rpt

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class detect_streettype:

    def __init__(self, ref, ref_name, downsample):
        self.ref = ref
        self.ref_name = ref_name
        self.downsample = downsample

    def detect_cp_offline(self, signal):
        """ detect changepoints in signal using PELT"""

        # filter signal and downsample
        points = moving_average(signal, self.downsample)[::self.downsample]

        # detect changpoints using PELT from ruptures
        model="rbf"
        algo = rpt.Pelt(model=model, min_size=300).fit(points)
        result = algo.predict(pen=1)
        self.result_upsample = [self.downsample*i for i in result] # updample results

        return self.result_upsample

    def classify(self, signal):
        """ classify """

        self.result_upsample = [0] + self.result_upsample
        pieces = []
        for i in range(0, len(self.result_upsample)-1):
            pieces.append(signal[self.result_upsample[i]:self.result_upsample[i+1]])

        class_list = []
        result_list = []

        for section in pieces:
            for num, reference in enumerate(self.ref):

                # TODO: better selection, could exceed range
                selected = section[100:100+len(reference)]

                # reference is longer than current signal
                if len(reference) > len(selected):
                    distance, _ = fastdtw(selected, reference[0:len(selected)], dist=euclidean)
                else:
                    distance, _ = fastdtw(selected, reference, dist=euclidean)
                
                if num == 0:
                    dist_min = distance
                    classification = self.ref_name[num]
                else:
                    if distance < dist_min:
                        dist_min = distance
                        classification = self.ref_name[num]

            class_list.append(classification)
            result_list = result_list + [classification] * len(section)
        
        return class_list, result_list






