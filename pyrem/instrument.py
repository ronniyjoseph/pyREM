import numpy as np
import sys
from scipy.constants import c


class RadioTelescope:

    def __init__(self, load=True, path=None, shape=None, frequency_channels=None, verbose=False):
        if verbose:
            print("Creating the radio telescope")
        self.antenna_id = None
        self.x_coordinate = None
        self.y_coordinate = None
        self.z_coordinate = None
        self.antenna_gain = None

        self.baseline_id = None
        self.u_coordinate = None
        self.v_coordinate = None
        self.w_coordinate = None
        self.antenna_id1 = None
        self.antenna_id2 = None
        self.number_of_baselines = None
        self.group_id = None
        self.selection = None


        #What's the hierarchy
        if shape is not None:
            data = create_xyz_positions(shape, verbose)
            self.antenna_id = data[:, 0]
            self.x_coordinate = data[:, 1]
            self.y_coordinate = data[:, 2]
            self.z_coordinate = data[:, 3]
        elif load:
            data = load_xyz_positions(path, verbose)
            self.antenna_id = data[:, 0]
            self.x_coordinate = data[:, 1]
            self.y_coordinate = data[:, 2]
            self.z_coordinate = data[:, 3]
        else:
            if verbose:
                print("Initialised empty class")
        return