import sys
import numpy
from .radiotelescope import BaselineTable
from .skymodel import apparent_fluxes_numba
from matplotlib import pyplot
sys.path.append("../../beam_perturbations/code/tile_beam_perturbations/")

from analytic_covariance import sky_covariance


def split_visibility(data):
    data_real = numpy.real(data)
    data_imag = numpy.imag(data)
    data_split = numpy.hstack((data_real, data_imag)).reshape((1, 2 * len(data_real)), order="C")
    return data_split[0, :]


def find_sky_model_sources(sky_realisation, frequency_range, antenna_size = 4, sky_model_depth = None):
    apparent_flux = apparent_fluxes_numba(sky_realisation, frequency_range, antenna_diameter = antenna_size )
    if sky_model_depth is None:
        rms = 3*numpy.sqrt(numpy.mean(sky_realisation.fluxes ** 2))
        sky_model_depth = rms

    model_source_indices = numpy.where(apparent_flux[:, 0] > sky_model_depth)
    sky_model = sky_realisation.select_sources(model_source_indices)

    return sky_model


def generate_sky_model_vectors(sky_model_sources, baseline_table, frequency_range, antenna_size, sorting_indices = None):
    number_of_sources = len(sky_model_sources.fluxes)
    sky_vectors = numpy.zeros((number_of_sources, 2*baseline_table.number_of_baselines))
    for i in range(number_of_sources):
        single_source = sky_model_sources.select_sources(i)
        source_visibilities = single_source.create_visibility_model(baseline_table, frequency_range, antenna_size)
        if sorting_indices is not None:
            sky_vectors[i, :] = split_visibility(source_visibilities[sorting_indices])

    return sky_vectors


def generate_covariance_vectors(number_of_baselines, frequency_range, sky_model_limit):
    covariance_vectors = numpy.zeros((2, number_of_baselines*2))
    covariance_vectors[0::2, 0::2] = 1
    covariance_vectors[1::2, 1::2] = 1
    covariance_vectors *= numpy.sqrt(sky_covariance(0, 0, frequency_range, S_high = sky_model_limit))
    return covariance_vectors


def hexagonal_array(nside):
    L = 14
    dL = 12
    antpos = []
    cen_y, cen_z = 0, 0
    for row in numpy.arange(nside):
        for cen_x in numpy.arange((2 * nside - 1) - row):
            dx = row / 2
            antpos.append(((cen_x + dx) * L, row * dL, cen_z))
            if row != 0:
                antpos.append(((cen_x + dx) * L, -row * dL, cen_z))

    return numpy.array(antpos)


def redundant_baseline_finder(baseline_table_object, group_minimum=3, threshold=1 / 6, verbose=False):
    """

    antenna_id1:
    antenna_id2:
    u:
    v:
    w:
    baseline_direction:
    group_minimum:
    threshold:
    verbose:
    :return:
    """
    antenna_id1 = baseline_table_object.antenna_id1
    antenna_id2 = baseline_table_object.antenna_id2

    u = baseline_table_object.u_coordinates
    v = baseline_table_object.v_coordinates
    w = baseline_table_object.w_coordinates

    isconj = (v < 0)& (u < 0)
    u = u.copy()
    v = v.copy()
    u[isconj] = -1 * u[isconj]
    v[isconj] = -1 * v[isconj]

    n_baselines = u.shape[0]
    # create empty table
    redundant_baselines = numpy.zeros((n_baselines, 6))
    # arbitrary counters
    # Let's find all the redundant baselines within our threshold
    group_counter = 0
    k = 0
    # Go through all antennas, take each antenna out and all antennas
    # which are part of the not redundant enough group

    while u.shape[0] > 0:
        # calculate uv separation at the calibration wavelength
        separation = numpy.sqrt((u - u[0]) ** 2. + (v - v[0]) ** 2.)
        # find all baselines within the lambda fraction
        select_indices = numpy.where(separation <= threshold)

        # is this number larger than the minimum number
        if len(select_indices[0]) >= group_minimum:
            # go through the selected baselines

            for i in range(len(select_indices[0])):
                # add antenna number
                redundant_baselines[k, 0] = antenna_id1[select_indices[0][i]]
                redundant_baselines[k, 1] = antenna_id2[select_indices[0][i]]
                # add coordinates uvw
                redundant_baselines[k, 2] = u[select_indices[0][i]]
                redundant_baselines[k, 3] = v[select_indices[0][i]]
                redundant_baselines[k, 4] = w[select_indices[0][i]]
                # add baseline group identifier
                redundant_baselines[k, 5] = 50000000 + 52 * (group_counter + 1)
                k += 1
            group_counter += 1
        # update the list, take out the used antennas
        all_indices = numpy.arange(len(u))
        unselected_indices = numpy.setdiff1d(all_indices, select_indices[0])

        antenna_id1 = antenna_id1[unselected_indices]
        antenna_id2 = antenna_id2[unselected_indices]
        u = u[unselected_indices]
        v = v[unselected_indices]
        w = w[unselected_indices]

    if verbose:
        print("There are", k, "redundant baselines in this array.")
        print("There are", group_counter, "redundant groups in this array")

    # find the filled entries
    non_zero_indices = numpy.where(redundant_baselines[:, 2] != 0)
    # remove the empty entries
    redundant_baselines = redundant_baselines[non_zero_indices[0], :]

    table_object = BaselineTable()
    table_object.antenna_id1 = redundant_baselines[:, 0]
    table_object.antenna_id2 = redundant_baselines[:, 1]
    table_object.u_coordinates = redundant_baselines[:, 2]
    table_object.v_coordinates = redundant_baselines[:, 3]
    table_object.w_coordinates = redundant_baselines[:, 4]
    table_object.group_indices = redundant_baselines[:, 5]

    table_object.reference_frequency = 150e6
    table_object.number_of_baselines = len(redundant_baselines[:, 0])

    return table_object

