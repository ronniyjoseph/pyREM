import numpy
import powerbox
from scipy import interpolate
from numba import prange, njit, float32, complex64, void
from .radiotelescope import ideal_gaussian_beam


class SkyRealisation:

    def __init__(self, sky_type, fluxes=0, l_coordinates=0, m_coordinates=0, spectral_indices=0,
                 seed=0, k1=4100, gamma1=1.59, k2=4100, gamma2=2.5, flux_low = 40e-3, flux_mid=1, flux_high=5.,
                 verbose=False):

        if verbose:
            print("Creating the sky realisation")

        if sky_type == "random":
            numpy.random.seed(seed)
            self.fluxes = stochastic_sky(seed, k1, gamma1, k2, gamma2, flux_low, flux_mid, flux_high)
            all_r = numpy.sqrt(numpy.random.uniform(0, 1, len(self.fluxes)))
            all_phi = numpy.random.uniform(0, 2. * numpy.pi, len(self.fluxes))

            self.l_coordinates = all_r * numpy.cos(all_phi)
            self.m_coordinates = all_r * numpy.sin(all_phi)
            self.spectral_indices = numpy.zeros_like(self.fluxes) + spectral_indices
        elif sky_type == "point":
            self.fluxes = check_type_convert_to_array(fluxes)
            self.l_coordinates = check_type_convert_to_array(l_coordinates)
            self.m_coordinates = check_type_convert_to_array(m_coordinates)
            self.spectral_indices = check_type_convert_to_array(spectral_indices)
        else:
            raise ValueError(f"sky_type must be 'random' or 'point' NOT {sky_type}")

        self.sky_image = None

        return

    def create_sky_image(self, frequency_channels, baseline_table=None, radiotelescope=None,
                         resolution=None, oversampling=2):

        #####################################
        # Assume the sky is flat
        #####################################

        source_flux = self.fluxes
        source_l = self.l_coordinates
        source_m = self.m_coordinates

        if type(frequency_channels) == numpy.ndarray:
            n_frequencies = len(frequency_channels)
        else:
            n_frequencies = 1

        if baseline_table is not None:
            # Find longest baseline to determine sky_image sampling, pick highest frequency for longest baseline
            max_u = numpy.max(numpy.abs(baseline_table.u(frequency_channels)))
            max_v = numpy.max(numpy.abs(baseline_table.v(frequency_channels)))
            max_b = max(max_u, max_v)
            # sky_resolutions
            min_l = 1. / (2 * max_b)
            delta_l = min_l / oversampling
        elif radiotelescope is not None:
            max_u = numpy.max(numpy.abs(radiotelescope.baseline_table.u(frequency_channels)))
            max_v = numpy.max(numpy.abs(radiotelescope.baseline_table.v(frequency_channels)))
            max_b = max(max_u, max_v)
            # sky_resolutions
            min_l = 1. / (2 * max_b)
            delta_l = min_l / oversampling
        elif resolution is not None:
            n_frequencies = 1
            delta_l = resolution / oversampling
        elif radiotelescope == None and resolution == None:
            raise ValueError("Input either a RadioTelescope object or specify a resolution")

        l_pixel_dimension = int(2. / delta_l)

        if l_pixel_dimension % 2 == 0:
            l_pixel_dimension += 1

        # empty sky_image
        if n_frequencies > 1:
            sky_image = numpy.zeros((l_pixel_dimension, l_pixel_dimension, n_frequencies))
        elif n_frequencies == 1:
            sky_image = numpy.zeros((l_pixel_dimension, l_pixel_dimension))

        l_coordinates = numpy.linspace(-1, 1, l_pixel_dimension)

        l_shifts = numpy.diff(l_coordinates) / 2.

        l_bin_edges = numpy.concatenate((numpy.array([l_coordinates[0] - l_shifts[0]]),
                                         l_coordinates[1:] - l_shifts,
                                         numpy.array([l_coordinates[-1] + l_shifts[-1]])))

        if n_frequencies > 1:
            for frequency_index in range(n_frequencies):
                sky_image[:, :, frequency_index], l_bins, m_bins = numpy.histogram2d(source_l, source_m,
                                                                                     bins=(l_bin_edges, l_bin_edges),
                                                                                     weights=source_flux)
        elif n_frequencies == 1:
            sky_image[:, :], l_bins, m_bins = numpy.histogram2d(source_l, source_m, bins=(l_bin_edges, l_bin_edges),
                                                                weights=source_flux)

        # normalise sky image for pixel size Jy/beam
        normalised_sky_image = sky_image / (2 / l_pixel_dimension) ** 2.

        return normalised_sky_image, l_coordinates

    def select_sources(self, indices):
        selected_fluxes = self.fluxes[indices]
        selected_l_coordinates = self.l_coordinates[indices]
        selected_m_coordinates = self.m_coordinates[indices]
        selected_spectral_indices = self.spectral_indices[indices]
        sky_selection = SkyRealisation(sky_type='point', fluxes=selected_fluxes, l_coordinates=selected_l_coordinates,
                                       m_coordinates=selected_m_coordinates, spectral_indices=selected_spectral_indices)

        return sky_selection

    def create_visibility_model(self, baseline_table_object, frequency_channels, antenna_size=4, mode='analytic', interpolation='spline',
                               padding_factor=3, parallel=False):

        if mode == 'analytic':
            visibilities = create_visibilities_analytic(self, baseline_table=baseline_table_object,
                                                        frequency_range=frequency_channels,
                                                        antenna_diameter=antenna_size)
        elif mode == 'numerical':
            visibilities = create_visibilities_numerical(self, baseline_table_object, frequency_channels,
                                                         antenna_size=antenna_size)

        return visibilities

    def save_table(self, path=None, filename=None):
        if path is None:
            path = "./"
        if filename is None:
            filename = "sky_catalogue"

        data = numpy.stack((self.fluxes, self.l_coordinates, self.m_coordinates, self.spectral_indices))
        numpy.save(path + filename, data)
        return


def stochastic_sky(seed=0, k1=4100, gamma1=1.59, k2=4100, \
                   gamma2=2.5, S_low=400e-3, S_mid=1, S_high=5.):
    numpy.random.seed(seed)

    # Franzen et al. 2016
    # k1 = 6998, gamma1 = 1.54, k2=6998, gamma2=1.54
    # S_low = 0.1e-3, S_mid = 6.0e-3, S_high= 400e-3 Jy

    # Cath's parameters
    # k1=4100, gamma1 =1.59, k2=4100, gamma2 =2.5
    # S_low = 0.400e-3, S_mid = 1, S_high= 5 Jy

    if S_low > S_mid:
        norm = k2 * (S_high ** (1. - gamma2) - S_low ** (1. - gamma2)) / (1. - gamma2)
        n_sources = numpy.random.poisson(norm * 2. * numpy.pi)
        # generate uniform distribution
        uniform_distr = numpy.random.uniform(size=n_sources)
        # initialize empty array for source fluxes
        source_fluxes = numpy.zeros(n_sources)
        source_fluxes = \
            (uniform_distr * norm * (1. - gamma2) / k2 +
             S_low ** (1. - gamma2)) ** (1. / (1. - gamma2))
    else:
        # normalisation
        norm = k1 * (S_mid ** (1. - gamma1) - S_low ** (1. - gamma1)) / (1. - gamma1) + \
               k2 * (S_high ** (1. - gamma2) - S_mid ** (1. - gamma2)) / (1. - gamma2)
        # transition between the one power law to the other
        mid_fraction = k1 / (1. - gamma1) * (S_mid ** (1. - gamma1) - S_low ** (1. - gamma1)) / norm
        n_sources = numpy.random.poisson(norm * 2. * numpy.pi)

        #########################
        # n_sources = 1e5
        #########################

        # generate uniform distribution
        uniform_distr = numpy.random.uniform(size=n_sources)
        # initialize empty array for source fluxes
        source_fluxes = numpy.zeros(n_sources)

        source_fluxes[uniform_distr < mid_fraction] = \
            (uniform_distr[uniform_distr < mid_fraction] * norm * (1. - gamma1) / k1 +
             S_low ** (1. - gamma1)) ** (1. / (1. - gamma1))

        source_fluxes[uniform_distr >= mid_fraction] = \
            ((uniform_distr[uniform_distr >= mid_fraction] - mid_fraction) * norm * (1. - gamma2) / k2 +
             S_mid ** (1. - gamma2)) ** (1. / (1. - gamma2))
    return source_fluxes


def check_type_convert_to_array(input_values):
    if type(input_values) != numpy.ndarray:
        converted = numpy.array([input_values])
    else:
        converted = input_values

    return converted


def create_visibilities_numerical(sky_realisation_object, baseline_table_object, frequency_channel, antenna_size):
    # check whether sky_image has already been created
    gridded_sky, l_coordinates = sky_realisation_object.create_sky_image(frequency_channel,
                                                       baseline_table=baseline_table_object)
    ll, mm = numpy.meshgrid(l_coordinates, l_coordinates)
    primary_beam = ideal_gaussian_beam(ll, mm, nu = frequency_channel, diameter=antenna_size)
    image_size = gridded_sky.shape[0]
    gridded_visibilities, uv_grid = powerbox.dft.fft(numpy.fft.ifftshift(numpy.pad(primary_beam*gridded_sky,
                                                                          (image_size*3,image_size*3), mode='constant')),
                                            L=2, axes=(0, 1))
    print(gridded_visibilities.shape)
    observations = uv_list_to_baseline_measurements(baseline_table_object, frequency_channel, gridded_visibilities, uv_grid)
    return


def uv_list_to_baseline_measurements(baseline_table_object, frequency_channels, visibility_grid, uv_grid):
    n_frequencies = len(frequency_channels)
    n_measurements = baseline_table_object.number_of_baselines
    # #First of all convert the uv_grid to a bin_edges array
    u_bin_size = numpy.median(numpy.diff(uv_grid[0]))
    v_bin_size = numpy.median(numpy.diff(uv_grid[1]))

    u_bin_centers = uv_grid[0] - u_bin_size / 2.
    v_bin_centers = uv_grid[1] - v_bin_size / 2.

    #now we have the bin edges we can start binning our baseline table
    #Create an empty array to store our baseline measurements in
    visibilities = numpy.zeros((n_measurements, n_frequencies), dtype=complex)

    for frequency_index in range(n_frequencies):
        visibility_data = visibility_grid

        real_component = interpolate.RegularGridInterpolator((u_bin_centers, v_bin_centers), numpy.real(visibility_data))
        imag_component = interpolate.RegularGridInterpolator((u_bin_centers, v_bin_centers), numpy.imag(visibility_data))


        visibilities[:, frequency_index] = real_component(baseline_table_object.u(frequency_channels),
                                                          baseline_table_object.v(frequency_channels[frequency_index])) + \
                                           1j*imag_component(baseline_table_object.u(frequency_channels[frequency_index]),
                                                          baseline_table_object.v(frequency_channels[frequency_index]))


        #u_index = numpy.digitize(baseline_table[:, 2, frequency_index], bins=u_bin_edges)
        #v_index = numpy.digitize(baseline_table[:, 3, frequency_index], bins=v_bin_edges)

        #print("centers in u bins", u_bin_centers[u_index-1]
        #visibilities[:, frequency_index] = visibility_grid[u_index, v_index,frequency_index]

    return visibilities


def create_visibilities_analytic(source_population, baseline_table, frequency_range, antenna_diameter=4):
    observations = numpy.zeros((baseline_table.number_of_baselines, len(frequency_range)), dtype=complex)

    # pre-compute all apparent fluxes at all frequencies
    apparent_flux = numpy.complex64(apparent_fluxes_numba(source_population, frequency_range, antenna_diameter))
    u_coordinates = baseline_table.u(frequency_range)
    v_coordinates = baseline_table.v(frequency_range)
    numba_loop(observations, apparent_flux, source_population.l_coordinates, source_population.m_coordinates,
               u_coordinates, v_coordinates)
    return observations


@njit(parallel=True)
def numba_loop(observations, fluxes, l_source, m_source, u_baselines, v_baselines):
    for baseline_index in prange(u_baselines.shape[0]):
        for source_index in range(len(fluxes)):
            for frequency_index in range(u_baselines.shape[1]):
                kernel = numpy.exp(-2j * numpy.pi *
                                   (u_baselines[baseline_index, frequency_index] * l_source[source_index] +
                                    v_baselines[baseline_index, frequency_index] * m_source[source_index]))
                observations[baseline_index, frequency_index] += fluxes[source_index, frequency_index] * kernel


def apparent_fluxes_numba(source_population, frequency_range, antenna_diameter=4):
    ff = numpy.tile(frequency_range, (len(source_population.fluxes), 1))
    ss = numpy.tile(source_population.fluxes, (len(frequency_range), 1))
    ll = numpy.tile(source_population.l_coordinates, (len(frequency_range), 1))
    mm = numpy.tile(source_population.m_coordinates, (len(frequency_range), 1))

    antenna_response = ideal_gaussian_beam(ll.T, mm.T, ff, diameter=antenna_diameter)

    apparent_fluxes = antenna_response * ss.T
    return apparent_fluxes


def sky_moment_returner(n_order, k1=4100, gamma1=1.59, k2=4100, gamma2=2.5, s_low=1e-5, s_mid=1, s_high=10.):
    # Check whether the breakpoints are correct
    if s_high < s_mid:
        s_mid = s_high
    if s_low > s_mid:
        s_mid = s_low

    moment = k1 / (n_order + 1 - gamma1) * (s_mid ** (n_order + 1 - gamma1) - s_low ** (n_order + 1 - gamma1)) + \
             k2 / (n_order + 1 - gamma2) * (s_high ** (n_order + 1 - gamma2) - s_mid ** (n_order + 1 - gamma2))

    return moment