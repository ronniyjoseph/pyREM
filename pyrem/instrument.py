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



def create_xyz_positions(shape, verbose=False):
    """
    Generates an array lay-out defined by input parameters, returns
    x,y,z coordinates of each antenna in the array
    shape	: list of array parameters
    shape[0]	: string value 'square', 'hex', 'doublehex', 'linear'
        'square': produces a square array
            shape[1]: 1/2 side of the square in meters
            shape[2]: number of antennas along 1 side
            shape[3]: x position of square
            shape[4]: y position of square
        'hex': produces a hex array
        'doublehex': produces a double hex array
        'linear': produces a linear array
            shape[1]: x-outeredges of the array
            shape[2]: number of elements in the EW-linear array
    """

    if shape[0] == "square" or shape[0] == 'doublesquare':
        if verbose:
            print("")
            print("Creating x- y- z-positions of a square array")
        x_coordinates = np.linspace(-shape[1], shape[1], shape[2])
        y_coordinates = np.linspace(-shape[1], shape[1], shape[2])

        block1 = np.zeros((len(x_coordinates) * len(y_coordinates), 4))
        k = 0
        for i in range(len(x_coordinates)):
            for j in range(len(y_coordinates)):
                block1[k, 0] = 1001 + k
                block1[k, 1] = x_coordinates[i]
                block1[k, 2] = y_coordinates[j]
                block1[k, 3] = 0
                k += 1
        if shape[0] == 'square':
            block1[:, 1] += shape[3]
            block1[:, 2] += shape[4]
            xyz_coordinates = block1.copy()
        elif shape[0] == 'doublesquare':
            block2 = block1.copy()

            block2[:, 0] += 1000 + len(block1[:, 0])
            block2[:, 1] += shape[3]
            block2[:, 2] += shape[4]
            xyz_coordinates = np.vstack((block1, block2))

    elif shape[0] == 'hex' or shape[0] == 'doublehex':
        if verbose:
            print("")
            print("Creating x- y- z-positions of a " + shape[0] + " array")

        dx = shape[1]
        dy = dx * np.sqrt(3.) / 2.

        line1 = np.array([np.arange(4) * dx, np.zeros(4), np.zeros(4)]).transpose()

        # define the second line
        line2 = line1[0:3, :].copy()
        line2[:, 0] += dx / 2.
        line2[:, 1] += dy
        # define the third line
        line3 = line1[0:3].copy()
        line3[:, 1] += 2 * dy
        # define the fourth line
        line4 = line2[0:2, :].copy()
        line4[:, 1] += 2 * dy

        block1 = np.vstack((line1[1:], line2, line3, line4))

        block2 = np.vstack((line1[1:], line2, line3[1:], line4))
        block2[:, 0] *= -1

        block3 = np.vstack((line2, line3, line4))
        block3[:, 1] *= -1

        block4 = np.vstack((line2, line3[1:], line4))
        block4[:, 0] *= -1
        block4[:, 1] *= -1
        hex_block = np.vstack((block1, block2, block3, block4))

        if shape[0] == 'hex':
            if len(shape) != 4:
                raise ValueError(f"shape input to generate 'hex' array should contain 4 entries NOT {len(shape)}\n" +
                                 "['hex', horizontal minimum spacing, x centre coordinate, y centre coordinate")
            hex_block[:, 0] += shape[2]
            hex_block[:, 1] += shape[3]
            antenna_numbers = np.arange(len(hex_block[:, 0])) + 1001
            xyz_coordinates = np.vstack((antenna_numbers, hex_block.T)).T
        elif shape[0] == 'doublehex':
            if len(shape) != 6:
                raise ValueError(f"shape input to generate 'hex' array should contain 6 entries NOT {len(shape)}\n" +
                                 "['hex', horizontal minimum spacing, x centre hex1, y centre hex1, x centre hex2, y centre hex2]")

            antenna_numbers = np.arange(len(hex_block[:, 0])) + 1001
            first_hex = np.vstack((antenna_numbers, hex_block.T)).T

            second_hex = first_hex.copy()

            first_hex[:, 1] += shape[2]
            first_hex[:, 2] += shape[3]

            second_hex[:, 0] += 1000 + len(first_hex[:, 0])
            second_hex[:, 1] += shape[4]
            second_hex[:, 2] += shape[5]
            xyz_coordinates = np.vstack((first_hex, second_hex))
    elif shape[0] == 'linear':
        if verbose:
            print("")
            print("Creating x- y- z-positions of a " + str(shape[2]) + " element linear array")
        xyz_coordinates = np.zeros((shape[2], 4))
        xyz_coordinates[:, 0] = np.arange(shape[2]) + 1001
        if len(shape) == 3:
            xyz_coordinates[:, 1] = np.linspace(-shape[1], shape[1], shape[2])
        elif len(shape) == 4 and shape[3] == 'log':
            xyz_coordinates[:, 1] = np.logspace(1, np.log10(shape[1]), shape[2])
        else:
            pass

    return xyz_coordinates


def load_xyz_positions(path, verbose=False):
    if verbose:
        print(f"Loading antenna positions from {path}")

    antenna_data = np.loadtxt(path)

    # Check whether antenna ids are passed are in here
    if antenna_data.shape[1] != 4:
        antenna_ids = np.arange(1, antenna_data.shape[0] + 1, 1).reshape((antenna_data.shape[0], 1))
        antenna_data = np.hstack((antenna_ids, antenna_data))
    elif antenna_data.shape[1] > 4:
        raise ValueError(f"The antenna position file should only contain 4 columns: antenna_id, x, y, z. \n " +
                         f"This file contains {antenna_data.shape[1]} columns")

    ### Is this really necessary
    antenna_data = antenna_data[np.argsort(antenna_data[:, 0])]

    return antenna_data