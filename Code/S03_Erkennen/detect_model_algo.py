import numpy as np
from scipy import signal
from Code.S03_Erkennen.detect_model import detect_model
from Code.S03_Erkennen.setpoint_algo import setpoint_algo

class detect_model_algo(detect_model, setpoint_algo):
    """ child class that combines model-based detection and the algorithm that computes setpoint"""

    def __init__(self, timespan):
        self.timespan = timespan # timespan based on which dist_setpoint is calculated
        self.name = 'model algo ' + str(timespan)

        

            
            