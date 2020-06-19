import numpy
import copy
import os
from scipy.constants import c
from scipy.special import jv

class RadioTelescope:

    def __init__(self, load=True, path=None, shape=None, frequency_channels=None, verbose=False):
        if verbose:
            print("Creating the radio telescope")
        self.antenna_positions = None
        if shape is not None:
            self.antenna_positions = AntennaPositions(load =False, path = None, shape=shape, verbose=verbose)
        if load:
            self.antenna_positions = AntennaPositions(load=True, path=path, shape=None, verbose=verbose)
        if shape is not None or load:
            self.baseline_table = BaselineTable(position_table=self.antenna_positions,
                                                frequency_channels=frequency_channels, verbose=verbose)
        else:
            self.baseline_table = None
        return


class AntennaPositions:
    def __init__(self, load=True, path=None, shape=None, verbose=False):
        if load:
            if path == None:
                raise ValueError("Specificy the antenna position path if loading position data")
            else:
                antenna_data = xyz_position_loader(path)

        if shape is not None:
            antenna_data = xyz_position_creator(shape, verbose=verbose)

        if load or shape is not None:
            self.antenna_ids = antenna_data[:, 0]
            self.x_coordinates = antenna_data[:, 1]
            self.y_coordinates = antenna_data[:, 2]
            self.z_coordinates = antenna_data[:, 3]
        else:
            self.antenna_ids = None
            self.x_coordinates = None
            self.y_coordinates = None
            self.z_coordinates = None

        if self.antenna_ids is not None:
            self.antenna_gains = numpy.zeros(len(self.antenna_ids), dtype=complex) + 1 + 0j
        else:
            self.antenna_gains = None
        return

    def number_antennas(self):
        return len(self.antenna_ids)

    def save_position_table(self, path=None, filename=None):

        if path is None:
            path = "./"
        if filename is None:
            filename = "telescope_positions"

        data = numpy.stack((self.antenna_ids, self.x_coordinates, self.y_coordinates, self.z_coordinates))
        numpy.save(path + filename, data)
        return

    def save_gain_table(self, path=None, filename=None):

        if path is None:
            path = "./"
        if filename is None:
            filename = "telescope_gains"

        data = self.antenna_gains
        numpy.save(path + filename, data)
        return


class BaselineTable:
    def __init__(self, position_table=None, frequency_channels=None, verbose=False):
        self.antenna_id1 = None
        self.antenna_id2 = None
        self.u_coordinates = None
        self.v_coordinates = None
        self.w_coordinates = None
        self.reference_frequency = None
        self.number_of_baselines = None
        self.group_indices = None
        self.selection = None
        self.baseline_gains = None
        # update all attributes
        if position_table is not None:
            self.baseline_converter(position_table, frequency_channels, verbose)
        return

    def baseline_converter(self, position_table, frequency_channels=None, verbose=True):
        if verbose:
            print("")
            print("Converting xyz to uvw-coordinates")
        if frequency_channels is None:
            self.reference_frequency = 150e6
        elif type(frequency_channels) == numpy.ndarray:
            assert min(frequency_channels) > 1e6, "Frequency range is smaller 1 MHz, probably wrong units"
            self.reference_frequency = frequency_channels[0]
        elif numpy.isscalar(frequency_channels):
            assert frequency_channels > 1e6, "Frequency range is smaller 1 MHz, probably wrong units"
            self.reference_frequency = frequency_channels
        else:
            raise ValueError(f"frequency_channels should be 'numpy.ndarray', or scalar not type({self.reference_frequency})")

        # calculate the wavelengths of the adjacent channels
        reference_wavelength = c / self.reference_frequency
        # Count the number of antenna
        number_of_antenna = position_table.number_antennas()
        # Calculate the number of possible baselines
        self.number_of_baselines = int(0.5 * number_of_antenna * (number_of_antenna - 1.))

        # Create arrays for the baselines
        # baselines x Antenna1, Antenna2, u, v, w, gain product, phase sum x channels
        antenna_1 = numpy.zeros(self.number_of_baselines)
        antenna_2 = antenna_1.copy()

        u_coordinates = antenna_1.copy()
        v_coordinates = antenna_1.copy()
        w_coordinates = antenna_1.copy()
        baseline_gains = numpy.zeros((self.number_of_baselines, 1), dtype=complex)

        if verbose:
            print("")
            print("Number of antenna =", number_of_antenna)
            print("Total number of baselines =", self.number_of_baselines)

        # arbitrary counter to keep track of the baseline table
        k = 0

        for i in range(number_of_antenna):
            for j in range(i + 1, number_of_antenna):
                # save the antenna numbers in the uv table
                antenna_1[k] = position_table.antenna_ids[i]
                antenna_2[k] = position_table.antenna_ids[j]

                # rescale and write uvw to multifrequency baseline table
                u_coordinates[k] = (position_table.x_coordinates[i] - position_table.x_coordinates[
                    j]) / reference_wavelength
                v_coordinates[k] = (position_table.y_coordinates[i] - position_table.y_coordinates[
                    j]) / reference_wavelength
                w_coordinates[k] = (position_table.z_coordinates[i] - position_table.z_coordinates[
                    j]) / reference_wavelength
                if position_table.antenna_gains is None:
                    baseline_gains[k] = 1 + 0j
                else:
                    baseline_gains[k] = position_table.antenna_gains[i]*numpy.conj(position_table.antenna_gains[j])

                k += 1

        self.antenna_id1 = antenna_1
        self.antenna_id2 = antenna_2

        self.u_coordinates = u_coordinates
        self.v_coordinates = v_coordinates
        self.w_coordinates = w_coordinates

        self.baseline_gains = baseline_gains
        return

    def u(self, frequency=None):
        rescaled_u = rescale_baseline(self.u_coordinates, self.reference_frequency, frequency)
        selected_rescaled_u = select_baselines(rescaled_u, self.selection)

        return selected_rescaled_u

    def v(self, frequency=None):
        rescaled_v = rescale_baseline(self.v_coordinates, self.reference_frequency, frequency)
        selected_rescaled_v = select_baselines(rescaled_v, self.selection)

        return selected_rescaled_v

    def w(self, frequency=None):
        rescaled_w = rescale_baseline(self.w_coordinates, self.reference_frequency, frequency)
        selected_rescaled_w = select_baselines(rescaled_w, self.selection)

        return selected_rescaled_w

    def sub_table(self, baseline_selection_indices):
        subtable = copy.copy(self)
        subtable.number_of_baselines = len(baseline_selection_indices)

        subtable.antenna_id1 = self.antenna_id1[baseline_selection_indices]
        subtable.antenna_id2 = self.antenna_id2[baseline_selection_indices]
        subtable.u_coordinates = self.u_coordinates[baseline_selection_indices]
        subtable.v_coordinates = self.v_coordinates[baseline_selection_indices]
        subtable.w_coordinates = self.w_coordinates[baseline_selection_indices]

        subtable.baseline_gains= self.baseline_gains[baseline_selection_indices]
        if self.group_indices is not None:
            subtable.group_indices = self.group_indices[baseline_selection_indices]
        return subtable

    def save_table(self, path=None, filename=None):

        if path is None:
            path = "./"
        if filename is None:
            filename = "baseline_table"

        data = numpy.stack((self.antenna_id1, self.antenna_id2, self.u_coordinates, self.v_coordinates,
                            self.w_coordinates))
        numpy.save(path + filename, data)
        return


def beam_width(frequency =150e6, diameter=4, epsilon=0.42):
    sigma = epsilon * c / (frequency * diameter)
    width = numpy.sin(0.5 * sigma)
    return width


def airy_beam(theta, nu=150e6, diameter = 6):
    k = 2*numpy.pi*nu/c

    beam = (2*jv(1, k*diameter*numpy.sin(theta))/(k*diameter*numpy.sin(theta)))**2.
    return beam


def ideal_gaussian_beam(source_l, source_m, nu, diameter=4, epsilon=0.42):
    sigma = beam_width(nu, diameter, epsilon)

    beam_attenuation = numpy.exp(-(source_l ** 2. + source_m ** 2.) / (2 * sigma ** 2))

    return beam_attenuation


def broken_gaussian_beam(source_l, source_m, nu, faulty_dipole, diameter=4, epsilon=0.42, dx=1.1):
    wavelength = c / nu
    x_offsets, y_offsets = mwa_dipole_locations(dx)

    dipole_beam = ideal_gaussian_beam(source_l, source_m, nu, diameter / 4., epsilon=epsilon)
    ideal_tile_beam = ideal_gaussian_beam(source_l, source_m, nu, diameter)
    broken_beam = ideal_tile_beam - 1 / 16 * dipole_beam * numpy.exp(
        -2. * numpy.pi * 1j * (x_offsets[faulty_dipole] * numpy.abs(source_l) +
                               y_offsets[faulty_dipole] * numpy.abs(source_m)) / wavelength)

    return broken_beam


def simple_mwa_tile(theta, phi, target_theta=0, target_phi=0, frequency=150e6, weights=1):
    dipole_sep = 1.1  # meters
    x_offsets = numpy.array([-1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5, -1.5,
                             -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5], dtype=numpy.float32) * dipole_sep
    y_offsets = numpy.array([1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5,
                             -0.5, -0.5, -1.5, -1.5, -1.5, -1.5], dtype=numpy.float32) * dipole_sep
    z_offsets = numpy.zeros(x_offsets.shape)

    weights += numpy.zeros(x_offsets.shape)

    dipole_jones_matrix = ideal_gaussian_beam(numpy.sin(theta),0, nu=frequency, diameter=1)
    array_factor = get_array_factor(x_offsets, y_offsets, z_offsets, weights, theta, phi, target_theta, target_phi,
                                    frequency)

    tile_response = array_factor * dipole_jones_matrix
    tile_response /=tile_response.max()


    return tile_response


def ideal_mwa_beam_loader(theta, phi, frequency, load=True, verbose=False):
    if not load:
        if verbose:
            print("Creating the idealised MWA beam\n")
        ideal_beam = mwa_tile_beam(theta, phi, frequency=frequency)
        if not os.path.exists("beam_maps"):
            print("")
            print("Creating beam map folder locally!")
            os.makedirs("beam_maps")
        numpy.save(f"beam_maps/ideal_beam_map.npy", ideal_beam)
    if load:
        if verbose:
            print("Loading the idealised MWA beam\n")
        ideal_beam = numpy.load(f"beam_maps/ideal_beam_map.npy")

    return ideal_beam


def broken_mwa_beam_loader(theta, phi, frequency, faulty_dipole = None, load=True):
    dipole_weights = numpy.zeros(16) + 1
    if faulty_dipole is not None:
        dipole_weights[faulty_dipole] = 0
    if load:
        print(f"Loading perturbed tile beam for dipole {faulty_dipole}")
        perturbed_beam = numpy.load(f"beam_maps/perturbed_dipole_{faulty_dipole}_map.npy")
    elif not load:
        # print(f"Generating perturbed tile beam for dipole {faulty_dipole}")
        perturbed_beam = mwa_tile_beam(theta, phi, weights=dipole_weights, frequency=frequency)
        if not os.path.exists("beam_maps"):
            print("")
            print("Creating beam map folder locally!")
            os.makedirs("beam_maps")
        numpy.save(f"beam_maps/perturbed_dipole_{faulty_dipole}_map.npy", perturbed_beam)

    return perturbed_beam


def rescale_baseline(baseline_coordinates, reference_frequency, frequency):
    if frequency is None:
        rescaled_coordinates = baseline_coordinates
    elif numpy.isscalar(frequency):
        rescaling_factor = frequency / reference_frequency
        rescaled_coordinates = baseline_coordinates * rescaling_factor
    elif type(frequency) == numpy.ndarray:
        rescaling_factor = frequency / reference_frequency
        coordinate_mesh, rescale_mesh = numpy.meshgrid(rescaling_factor, baseline_coordinates)
        rescaled_coordinates = coordinate_mesh * rescale_mesh
    else:
        raise ValueError(f"frequency should be scalar or numpy.ndarray not {type(frequency)}")

    return rescaled_coordinates


def select_baselines(baseline_coordinates, baseline_selection_indices):
    if baseline_selection_indices is None:
        selected_baseline_coordinates = baseline_coordinates
    else:
        selected_baseline_coordinates = baseline_coordinates[baseline_selection_indices, ...]
    return selected_baseline_coordinates


def mwa_tile_beam(theta, phi, target_theta=0, target_phi=0, frequency=150e6, weights=1, dipole_type='cross',
                  gaussian_width=30 / 180 * numpy.pi):
    dipole_sep = 1.1  # meters
    x_offsets, y_offsets = mwa_dipole_locations(dipole_sep)
    z_offsets = numpy.zeros(x_offsets.shape)

    weights += numpy.zeros(x_offsets.shape)

    if dipole_type == 'cross':
        dipole_jones_matrix = cross_dipole(theta)
    elif dipole_type == 'gaussian':
        # print(theta_width)
        dipole_jones_matrix = gaussian_response(theta, gaussian_width)
    else:
        print("Wrong dipole_type: select cross or gaussian")

    ground_plane_field = electric_field_ground_plane(theta, frequency)
    array_factor = get_array_factor(x_offsets, y_offsets, z_offsets, weights, theta, phi, target_theta, target_phi,
                                    frequency)

    tile_response = array_factor * ground_plane_field * dipole_jones_matrix
    tile_response[numpy.isnan(tile_response)] = 0

    if len(theta.shape) > 2:
        beam_normalisation = numpy.add(numpy.zeros(tile_response.shape), numpy.amax(tile_response, axis=(0, 1)))
    else:
        beam_normalisation = numpy.add(numpy.zeros(tile_response.shape), numpy.amax(tile_response))
    normalised_response = tile_response / beam_normalisation * numpy.sum(weights) / 16

    return normalised_response


def get_array_factor(x, y, z, weights, theta, phi, theta_pointing=0, phi_pointing=0, frequency=150e6):
    wavelength = c / frequency
    number_dipoles = len(x)

    k_x = (2. * numpy.pi / wavelength) * numpy.sin(theta) * numpy.sin(phi)
    k_y = (2. * numpy.pi / wavelength) * numpy.sin(theta) * numpy.cos(phi)
    k_z = (2. * numpy.pi / wavelength) * numpy.cos(theta)

    k_x0 = (2. * numpy.pi / wavelength) * numpy.sin(theta_pointing) * numpy.sin(phi_pointing)
    k_y0 = (2. * numpy.pi / wavelength) * numpy.sin(theta_pointing) * numpy.cos(phi_pointing)
    k_z0 = (2. * numpy.pi / wavelength) * numpy.cos(theta_pointing)
    array_factor_map = numpy.zeros(theta.shape, dtype=complex)

    for i in range(number_dipoles):
        complex_exponent = -1j * ((k_x - k_x0) * x[i] + (k_y - k_y0) * y[i] + (k_z - k_z0) * z[i])

        # !This step takes a long time, look into optimisation through vectorisation/clever numpy usage
        dipole_factor = weights[i] * numpy.exp(complex_exponent)

        array_factor_map += dipole_factor

    # filter all NaN
    array_factor_map[numpy.isnan(array_factor_map)] = 0
    array_factor_map = array_factor_map / numpy.sum(weights)

    return array_factor_map


def electric_field_ground_plane(theta, frequency=150e6, height=0.3):
    wavelength = c / frequency
    ground_plane_electric_field = numpy.sin(2. * numpy.pi * height / wavelength * numpy.cos(theta))
    return ground_plane_electric_field


def cross_dipole(theta):
    response = numpy.cos(theta)
    return response


def xyz_position_loader(path):
    antenna_data = numpy.loadtxt(path)
    # Check whether antenna ids are passed are in here
    if antenna_data.shape[1] != 4:
        antenna_ids = numpy.arange(1, antenna_data.shape[0] + 1, 1).reshape((antenna_data.shape[0], 1))
        antenna_data = numpy.hstack((antenna_ids, antenna_data))
    elif antenna_data.shape[1] > 4:
        raise ValueError(f"The antenna position file should only contain 4 columns: antenna_id, x, y, z. \n " +
                         f"This file contains {antenna_data.shape[1]} columns")

    antenna_data = antenna_data[numpy.argsort(antenna_data[:, 0])]

    return antenna_data


def xyz_position_creator(shape, verbose=False):
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
        x_coordinates = numpy.linspace(-shape[1], shape[1], shape[2])
        y_coordinates = numpy.linspace(-shape[1], shape[1], shape[2])

        block1 = numpy.zeros((len(x_coordinates) * len(y_coordinates), 4))
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
            xyz_coordinates = numpy.vstack((block1, block2))

    elif shape[0] == 'hex' or shape[0] == 'doublehex':
        if verbose:
            print("")
            print("Creating x- y- z-positions of a " + shape[0] + " array")

        dx = shape[1]
        dy = dx * numpy.sqrt(3.) / 2.

        line1 = numpy.array([numpy.arange(4) * dx, numpy.zeros(4), numpy.zeros(4)]).transpose()

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

        block1 = numpy.vstack((line1[1:], line2, line3, line4))

        block2 = numpy.vstack((line1[1:], line2, line3[1:], line4))
        block2[:, 0] *= -1

        block3 = numpy.vstack((line2, line3, line4))
        block3[:, 1] *= -1

        block4 = numpy.vstack((line2, line3[1:], line4))
        block4[:, 0] *= -1
        block4[:, 1] *= -1
        hex_block = numpy.vstack((block1, block2, block3, block4))

        if shape[0] == 'hex':
            if len(shape) != 4:
                raise ValueError(f"shape input to generate 'hex' array should contain 4 entries NOT {len(shape)}\n" +
                                 "['hex', horizontal minimum spacing, x centre coordinate, y centre coordinate")
            hex_block[:, 0] += shape[2]
            hex_block[:, 1] += shape[3]
            antenna_numbers = numpy.arange(len(hex_block[:, 0])) + 1001
            xyz_coordinates = numpy.vstack((antenna_numbers, hex_block.T)).T
        elif shape[0] == 'doublehex':
            if len(shape) != 6:
                raise ValueError(f"shape input to generate 'hex' array should contain 6 entries NOT {len(shape)}\n" +
                                 "['hex', horizontal minimum spacing, x centre hex1, y centre hex1, x centre hex2, y centre hex2]")

            antenna_numbers = numpy.arange(len(hex_block[:, 0])) + 1001
            first_hex = numpy.vstack((antenna_numbers, hex_block.T)).T

            second_hex = first_hex.copy()

            first_hex[:, 1] += shape[2]
            first_hex[:, 2] += shape[3]

            second_hex[:, 0] += 1000 + len(first_hex[:, 0])
            second_hex[:, 1] += shape[4]
            second_hex[:, 2] += shape[5]
            xyz_coordinates = numpy.vstack((first_hex, second_hex))
    elif shape[0] == 'linear':
        if verbose:
            print("")
            print("Creating x- y- z-positions of a " + str(shape[2]) + " element linear array")
        xyz_coordinates = numpy.zeros((shape[2], 4))
        xyz_coordinates[:, 0] = numpy.arange(shape[2]) + 1001
        if len(shape) == 3:
            xyz_coordinates[:, 1] = numpy.linspace(-shape[1], shape[1], shape[2])
        elif len(shape) == 4 and shape[3] == 'log':
            xyz_coordinates[:, 1] = numpy.logspace(1, numpy.log10(shape[1]), shape[2])
        else:
            pass

    return xyz_coordinates


def redundant_baseline_finder(uv_positions, baseline_direction, verbose=False, minimum_baselines = 3,
                              wave_fraction = 1. / 6 ):
    """
	"""

    ################################################################

    ################################################################

    n_baselines = uv_positions.shape[0]
    n_frequencies = uv_positions.shape[2]
    middle_index = (n_frequencies + 1) // 2 - 1
    # create empty table
    baseline_selection = numpy.zeros((n_baselines, 8, n_frequencies))
    # arbitrary counters
    # Let's find all the redundant baselines within our threshold
    group_counter = 0
    k = 0
    # Go through all antennas, take each antenna out and all antennas
    # which are part of the not redundant enough group
    while uv_positions.shape[0] > 0:
        # calculate uv separation at the calibration wavelength
        separation = numpy.sqrt(
            (uv_positions[:, 2, middle_index] - uv_positions[0, 2, middle_index]) ** 2. +
            (uv_positions[:, 3, middle_index] - uv_positions[0, 3, middle_index]) ** 2.)
        # find all baselines within the lambda fraction
        select_indices = numpy.where(separation <= wave_fraction)

        # is this number larger than the minimum number
        if len(select_indices[0]) >= minimum_baselines:
            # go through the selected baselines

            for i in range(len(select_indices[0])):
                # add antenna number
                baseline_selection[k, 0, :] = uv_positions[select_indices[0][i], 0, :]
                baseline_selection[k, 1, :] = uv_positions[select_indices[0][i], 1, :]
                # add coordinates uvw
                baseline_selection[k, 2, :] = uv_positions[select_indices[0][i], 2, :]
                baseline_selection[k, 3, :] = uv_positions[select_indices[0][i], 3, :]
                baseline_selection[k, 4, :] = uv_positions[select_indices[0][i], 4, :]
                # add the gains
                baseline_selection[k, 5, :] = uv_positions[select_indices[0][i], 5, :]
                baseline_selection[k, 6, :] = uv_positions[select_indices[0][i], 6, :]
                # add baseline group identifier
                baseline_selection[k, 7, :] = 50000000 + 52 * (group_counter + 1)

                k += 1
            group_counter += 1
        # update the list, take out the used antennas
        all_indices = numpy.arange(len(uv_positions))
        unselected_indices = numpy.setdiff1d(all_indices, select_indices[0])

        uv_positions = uv_positions[unselected_indices]

    if verbose:
        print("There are", k, "redundant baselines in this array.")
        print("There are", group_counter, "redundant groups in this array")

    # find the filled entries
    non_zero_indices = numpy.where(baseline_selection[:, 0, 0] != 0)
    # remove the empty entries
    baseline_selection = baseline_selection[non_zero_indices[0], :, :]
    # Sort on length
    baseline_lengths = numpy.sqrt(baseline_selection[:, 2, middle_index] ** 2 \
                                  + baseline_selection[:, 3, middle_index] ** 2)

    sorted_baselines = baseline_selection[numpy.argsort(baseline_lengths), :, :]

    sorted_baselines = baseline_selection[numpy.argsort(sorted_baselines[:, 7, middle_index]), :, :]
    # sorted_baselines = sorted_baselines[numpy.argsort(sorted_baselines[:,1,middle_index]),:,:]
    # if we want only the EW select all the  uv positions around v = 0
    if baseline_direction == "EW":
        ew_indices = numpy.where(abs(sorted_baselines[:, 3, middle_index]) < 5. / wavelength)
        selected_baselines = sorted_baselines[ew_indices[0], :, :]
    elif baseline_direction == "NS":
        ns_indices = numpy.where(abs(sorted_baselines[:, 2, middle_index]) < 5. / wavelength)
        selected_baselines = sorted_baselines[ns_indices[0], :, :]
    elif baseline_direction == "ALL":
        selected_baselines = sorted_baselines
    else:
        sys.exit("The given redundant baseline direction is invalid:" + \
                 " please use 'EW', 'ALL'")
    return sorted_baselines


def mwa_dipole_locations(dx = 1):
    x = numpy.array([-1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5, -1.5,
                             -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5], dtype=numpy.float32) * dx

    y = numpy.array([1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5,
                             -0.5, -0.5, -1.5, -1.5, -1.5, -1.5], dtype=numpy.float32) * dx
    return x, y