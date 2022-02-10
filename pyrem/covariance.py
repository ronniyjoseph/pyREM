import numpy as np
from scipy.constants import c
from scipy import signal

from .radiotelescope import beam_width
from .radiotelescope import mwa_dipole_locations

from .skymodel import sky_moment_returner
from .powerspectrum import compute_power

class CovarianceMatrix:

    #Currently only per baseline not per real and imaginary component of each baseline

    def __init__(self, u, v, nu, group_ids, per_frequency=True):
        self.u = u
        self.v = v
        self.nu = nu
        self.group_ids = group_ids
        self.beam_model =  None
        self.baselines =  None
        self.matrix =  None
        self.eigenvalues = None
        self.eigenmodes =  None
        self.per_frequency_flag = per_frequency

        return

    def gaussian(self, aperture_diameter=2, S_low=1e-3, S_mid=1, S_high=10, gamma=0.8, nu_0 = 150e6):
        """
        Flat-sky gaussian approximation of a antenna beam

        """
        #Compute 2nd moment of the Point Source counts function thingy

        #So this point source sky model is calibrated for 150 MHz, can we scale this?
        u, v, nu =  self.return_grid()
        mu = sky_moment_returner(2, s_low=S_low, s_mid=S_mid, s_high=S_high)

        #If we're computing the covariance matrix over multiple frequencies we still want to end up with a 2D array that
        #that has the baseline-frequency covariance

        beam_width1 = beam_width(nu[0], diameter=aperture_diameter)
        beam_width2 = beam_width(nu[1], diameter=aperture_diameter)
        sigma_nu = beam_width1 ** 2 * beam_width2 ** 2 / (beam_width1 ** 2 + beam_width2 ** 2)

        kernel = -2 * np.pi ** 2 * sigma_nu * ( (u[0] * nu[0] - u[1] * nu[1]) ** 2 +
                                                (v[0] * nu[0] - v[1] * nu[1]) ** 2) / nu_0 ** 2
        covariance = 2 * np.pi * mu * sigma_nu * (nu[0] * nu[1] / nu_0 ** 2) ** (- gamma) * np.exp(kernel)
        self.matrix = covariance
        return covariance

    def return_grid(self):
        #work in an option to do full frequency/and per frequency because it would be good to be able and look at
        #the evolution of the eigenmodes as a function of frequency easily

        uu1, uu2 = np.meshgrid(self.u, self.u)
        vv1, vv2 = np.meshgrid(self.v, self.v)

        u_grid1 = np.tile(uu1, np.array([len(self.nu), len(self.nu)]))
        u_grid2 = np.tile(uu2, np.array([len(self.nu), len(self.nu)]))

        v_grid1 = np.tile(vv1, np.array([len(self.nu), len(self.nu)]))
        v_grid2 = np.tile(vv2, np.array([len(self.nu), len(self.nu)]))

        nu_grid1 = np.zeros_like(u_grid1)
        nu_grid2 = np.zeros_like(u_grid1)

        # Create the full covariance_coordinate grid?
        # if frequency flag is true, create a grid to store the per frequency channel covariance matrix


        for i in range(len(self.nu)):
            nu_grid1[i * len(self.u):(i + 1) * len(self.u), :] = self.nu[i]
            nu_grid2[:, i * len(self.u):(i + 1) * len(self.u)] = self.nu[i]

        return (u_grid1, u_grid2), (v_grid1, v_grid2), (nu_grid1, nu_grid2)


    def eigendecomposition(self, number=3, tolerance=1e-5):
        #Stores the first 'number' eigenmodes within tolerance level of the highest amplitude

        #Determine the number of unique groups + all the non grouped baselines
        group_ids = np.unique(self.group_ids)
        #remove non-redundant baselines
        #initialise an array to keep the eigenvalues per redundant group and per selected mode
        self.eigenvalues = np.zeros((len(group_ids), number), dtype=complex)
        #initialise an array to store the eigenvectors (vertically stakced) per redundant group and selected mode
        self.eigenmodes = np.zeros((len(self.group_ids), number), dtype=complex)

        #iterate over each redundant group
        for i in range(len(group_ids)):
            if group_ids[i] > 0:
                #Find all baselines that belong to this group
                index =  np.where(self.group_ids == group_ids[i])[0]
                # check whether they are contiguous, if not the data may need to be re-sorted
                if np.diff(index).max() > 1:
                    print("Warning: (quasi)-redundant baselines may not be organised in groups")
                #Create 2D selection indices/could also slice, but this will work when data is non contiguous
                ii, jj = np.meshgrid(index, index)
                #Select the block component from the covariance matrix and reshape into a NxN matrix
                block_matrix = self.matrix[ii, jj].reshape((index.shape[0],index.shape[0]))
                #Decompose into eigemodes
                eigenvalues, eigenmodes = np.linalg.eig(block_matrix)

                #Find eigenmodes within the selected tolerance
                eigen_index = np.where(eigenvalues > tolerance*eigenvalues.max())[0]
                #Sort the selected indices by the eigenvalue amplitudes
                eigen_index = eigen_index[np.argsort(np.abs(eigenvalues[eigen_index]))]

                #Need to figure out whether this number if larger or smaller than requested eigenmodes
                #Also would be nice to sort the eigenmodes

                if len(eigen_index) < number:
                    n_modes = len(eigen_index)
                else:
                    n_modes = number
                self.eigenvalues[i, :n_modes] = eigenvalues[eigen_index[:n_modes]]
                self.eigenmodes[index, :n_modes] = eigenmodes[:, eigen_index[:n_modes]]

        return self.eigenvalues, self.eigenmodes


    def reconstruct_matrix(self):
        matrix =  np.zeros((len(self.group_ids), len(self.group_ids)), dtype=complex)
        group_ids = np.unique(self.group_ids)
        for i in range(len(group_ids)):
            index = np.where(self.group_ids == group_ids[i])[0]

            eigenvalues = self.eigenvalues[i, :]
            eigenmodes = self.eigenmodes[index, :]
            matrix[index[0]:index[-1]+1,index[0]:index[-1]+1] = eigenmodes.dot(np.diag(eigenvalues)).dot(np.conj(eigenmodes).T)
        return matrix

    def airy(self):
        #So we can compute the Fourier Transform of the Airy Power beam pattern, however, for the cross covariance between
        #two different baselines we need to compute the the convolution between this functions. This looks to be a pain,
        #so instead we compute that part numerically in absence of an analytical solution.
        return

    def actualbeam(self):
        return

    def compute(self):
        return

    def decompose(self):
        return

def position_covariance(u, v, nu, position_precision = 1e-2, gamma = 0.8, mode = "frequency", nu_0 = 150e6,
                        tile_diameter = 4, s_high = 10):
    mu_1 = sky_moment_returner(n_order = 1, s_high= s_high)
    mu_2 = sky_moment_returner(n_order = 2, s_high= s_high)
    if mode == "frequency":
        nn1, nn2 = np.meshgrid(nu, nu)
        vv1 = v
        vv2 = v
        uu1 = u
        uu2 = u
        delta_u = position_precision*nu[0] / c
    else:
        nn1 = nu
        nn2 = nu
        vv1, vv2 = np.meshgrid(v, v)
        uu1, uu2 = np.meshgrid(u, u)
        delta_u = position_precision*nu / c

    beamwidth1 = beam_width(nn1, diameter=tile_diameter)
    beamwidth2 = beam_width(nn2, diameter=tile_diameter)

    sigma = beamwidth1**2*beamwidth2**2/(beamwidth1**2 + beamwidth2**2)

    kernel = -2*np.pi**2*sigma*((uu1*nn1 - uu2*nn2)**2 + (vv1*nn1 - vv2*nn2)**2 )/nu_0**2
    a = 16*np.pi**3*mu_2*(nn1*nn2/nu_0**2)**(1-gamma)*delta_u**2*sigma*np.exp(kernel)*(1+2*kernel)
    b = mu_1**2*(nn1*nn2)**(-gamma)*delta_u**2

    covariance = a
    return covariance


def beam_covariance(u, v, nu, dx = 1.1, gamma= 0.8, mode = 'frequency', broken_tile_fraction = 1.0, nu_0 = 150e6,
                    calibration_type = "sky", tile_diameter = 4, model_limit = 1):


    x_offsets, y_offsets = mwa_dipole_locations(dx)
    mu_1_r = sky_moment_returner(1, s_low = 1e-5,  s_high=model_limit)
    mu_2_r = sky_moment_returner(2, s_low = 1e-5, s_high=model_limit)

    mu_1_m = sky_moment_returner(1, s_low=model_limit, s_high=10)
    mu_2_m = sky_moment_returner(2, s_low=model_limit, s_high=10)

    if mode == "frequency":
        nn1, nn2, xx = np.meshgrid(nu, nu, x_offsets)
        nn1, nn2, yy = np.meshgrid(nu, nu, y_offsets)
        vv1 = v
        vv2 = v
        uu1 = u
        uu2 = u
        frequency_scaling = nn1[..., 0]*nn2[..., 0]/nu_0**2
    else:
        nn1 = nu
        nn2 = nu
        vv1, vv2, yy = np.meshgrid(v, v, y_offsets)
        uu1, uu2, xx = np.meshgrid(u, u, x_offsets)
        frequency_scaling = nu**2/nu_0**2

    width_1_tile = np.sqrt(2) * beam_width(frequency=nn1, diameter=tile_diameter)
    width_2_tile = np.sqrt(2) * beam_width(frequency=nn2, diameter=tile_diameter)
    width_1_dipole = np.sqrt(2) * beam_width(frequency=nn1, diameter=1)
    width_2_dipole = np.sqrt(2) * beam_width(frequency=nn2, diameter=1)

    kernel = -2 * np.pi ** 2 * ((uu1*nn1 - uu2*nn2 + xx*(nn1 - nn2) / c) ** 2 +
                                             (vv1*nn1 - vv2*nn2 + yy*(nn1 - nn2) / c) ** 2)/nu_0**2

    sigma_a = (width_1_tile * width_2_tile * width_1_dipole * width_2_dipole) ** 2 / (
            width_2_tile ** 2 * width_1_dipole ** 2 * width_2_dipole ** 2 +
            width_1_tile ** 2 * width_1_dipole ** 2 * width_2_dipole ** 2 +
            width_1_tile ** 2 * width_2_tile ** 2 * width_1_dipole ** 2 +
            width_1_tile ** 2 * width_2_tile ** 2 * width_2_dipole ** 2)

    sigma_b = (width_1_tile * width_2_tile * width_2_dipole) ** 2 / (
            2 * width_2_tile ** 2 * width_2_dipole ** 2 + width_1_tile ** 2 * width_2_dipole ** 2 +
            width_1_tile ** 2 * width_2_tile ** 2)

    sigma_c = (width_1_tile * width_2_tile * width_1_dipole) ** 2 / (
            width_2_tile ** 2 * width_1_dipole ** 2 + 2 * width_1_tile ** 2 * width_1_dipole ** 2 +
            width_1_tile ** 2 * width_2_tile ** 2)

    sigma_d1 = width_1_tile ** 2 * width_1_dipole ** 2 / (width_1_tile ** 2 + width_1_dipole ** 2)
    sigma_d2 = width_2_tile ** 2 * width_2_dipole ** 2 / (width_2_tile ** 2 + width_2_dipole ** 2)

    covariance_a = 2 * np.pi *frequency_scaling**(-gamma)* (mu_2_m + mu_2_r) /len(y_offsets) ** 3 * \
        np.sum(sigma_a * np.exp(sigma_a*kernel), axis=-1)

    covariance_b = -2 * np.pi *frequency_scaling**(-gamma)* mu_2_r / len(y_offsets) ** 2 * \
        np.sum(sigma_b * np.exp(sigma_b*kernel), axis=-1)

    covariance_c = -2 * np.pi *frequency_scaling**(-gamma)* mu_2_r / len(y_offsets) ** 2 * \
        np.sum(sigma_c * np.exp(sigma_c *kernel),axis=-1)

    covariance_d = 2 * np.pi *frequency_scaling**(-gamma)* (mu_1_m + mu_1_r)**2 * \
        np.sum( sigma_d1 * sigma_d2 / len(x_offsets) ** 3 * \
        np.exp(sigma_d1 * kernel)* np.exp(sigma_d2 * kernel), axis=-1)

    covariance_e = -2 * np.pi *frequency_scaling**(-gamma)*(mu_1_m + mu_1_r)**2 * \
        np.sum(sigma_d1 * sigma_d2 / len(x_offsets) ** 4 * np.exp(sigma_d1 * kernel), axis=-1) * \
        np.sum(np.exp(sigma_d2 * kernel), axis=-1)

    if calibration_type == "redundant":
        covariance = broken_tile_fraction**2*(covariance_a + covariance_d + covariance_e)
    if calibration_type == "sky":
        covariance = broken_tile_fraction**2*(covariance_a + covariance_b + covariance_c + covariance_d + covariance_e)

    return covariance


def sky_covariance(u, v, nu, S_low=1e-3, S_mid=1, S_high=10, gamma=0.8, mode = 'frequency', nu_0 = 150e6,
                   tile_diameter=4):

    mu_2 = sky_moment_returner(2, s_low=S_low, s_mid=S_mid, s_high=S_high)
    if mode == "frequency":
        nn1, nn2 = np.meshgrid(nu, nu)
        uu1 = u
        uu2 = u
        vv1 = v
        vv2 = v
    else:
        nn1 = nu
        nn2 = nu
        uu1, uu2 = np.meshgrid(u, u)
        vv1, vv2 = np.meshgrid(v, v)

    width_tile1 = beam_width(nn1, diameter=tile_diameter)
    width_tile2 = beam_width(nn2, diameter=tile_diameter)
    sigma_nu = width_tile1**2*width_tile2**2/(width_tile1**2 + width_tile2**2)


    kernel = -2*np.pi ** 2 * sigma_nu * ((uu1*nn1 - uu2*nn2) ** 2 + (vv1*nn1 - vv2*nn2) ** 2)/nu_0**2
    covariance = 2 * np.pi * mu_2 * sigma_nu * (nn1*nn2/nu_0**2)**(-gamma)*np.exp(kernel)

    return covariance


def thermal_variance(sefd=20e3, bandwidth=40e3, t_integrate=120):
    variance = (sefd / np.sqrt(bandwidth * t_integrate))**2

    return variance



def gain_error_covariance(u_range, frequency_range, residuals='both', weights=None, broken_baseline_weight=1,
                          calibration_type = 'sky', N_antenna = 128, tile_diameter = 4, position_error = 0.02,
                          model_limit=1):

    if calibration_type == "sky":
        model_variance = np.diag(sky_covariance(0, 0, frequency_range, S_low=model_limit, S_high=10,
                                                   tile_diameter=tile_diameter))
    elif calibration_type == 'relative':
        model_variance = np.diag(sky_covariance(0, 0, frequency_range, S_low=1e-3, S_high=10,
                                                   tile_diameter=tile_diameter))

    model_normalisation = np.sqrt(np.outer(model_variance, model_variance))


    covariance = np.zeros((len(u_range), len(frequency_range), len(frequency_range)))

    # Compute all residual to model ratios at different u scales
    for u_index in range(len(u_range)):
        if calibration_type == "sky":
            if residuals == 'sky':
                residual_covariance = sky_covariance(u_range[u_index], v=0, nu=frequency_range, S_high=model_limit,
                                                     tile_diameter=tile_diameter)
            elif residuals == "beam":
                residual_covariance = beam_covariance(u_range[u_index], v=0, nu=frequency_range, calibration_type ='sky',
                                                      broken_tile_fraction=broken_baseline_weight,
                                                      tile_diameter=tile_diameter, model_limit=model_limit)
            elif residuals == "both":
                residual_covariance = sky_covariance(u_range[u_index], v=0, nu=frequency_range,
                                                     tile_diameter=tile_diameter, S_high=model_limit) + \
                                      beam_covariance(u_range[u_index], v=0, nu=frequency_range, calibration_type ='sky',
                                                      broken_tile_fraction=broken_baseline_weight,
                                                      tile_diameter=tile_diameter, model_limit=model_limit)
            else:
                raise ValueError(f"{residuals} is an invalid residual option for calibration type {calibration_type}")
        elif calibration_type == 'relative':
            if residuals == 'position':
                residual_covariance = position_covariance(u_range[u_index], v=0, nu=frequency_range,
                                                          tile_diameter=tile_diameter, position_precision=position_error)
            elif residuals == "beam":
                residual_covariance = beam_covariance(u_range[u_index], v=0, nu=frequency_range,
                                                      tile_diameter=tile_diameter,
                                                      calibration_type ='redundant',
                                                      broken_tile_fraction=broken_baseline_weight)
            elif residuals == "both":
                residual_covariance = position_covariance(u_range[u_index], v=0, nu=frequency_range,
                                                          tile_diameter=tile_diameter, position_precision=position_error) + \
                                      beam_covariance(u_range[u_index], v=0, nu=frequency_range,
                                                      calibration_type='redundant',
                                                      broken_tile_fraction=broken_baseline_weight,
                                                      tile_diameter=tile_diameter)
            else:
                raise ValueError(f"{residuals} is an invalid residual option for calibration type {calibration_type}")


        covariance[u_index, :, :] = residual_covariance / model_normalisation

    if weights is None:
        gain_averaged_covariance = np.sum(covariance, axis=0) * (1/((N_antenna - 1)*len(u_range))) ** 2
    else:
        gain_averaged_covariance = covariance.copy()
        for u_index in range(len(u_range)):
            u_weight_reshaped = np.tile(weights[u_index, :].flatten(), (len(frequency_range), len(frequency_range), 1)).T
            gain_averaged_covariance[u_index, ...] = np.sum(covariance * u_weight_reshaped, axis=0)

    return gain_averaged_covariance


def compute_weights(u_cells, u, v, N_antenna = 128):
    u_bin_edges = np.zeros(len(u_cells) + 1)
    baseline_lengths = np.sqrt(u**2 + v**2)
    log_steps = np.diff(np.log10(u_cells))
    u_bin_edges[1:] = 10**(np.log10(u_cells) + 0.5*log_steps[0])
    u_bin_edges[0] = 10**(np.log10(u_cells[0] - 0.5*log_steps[0]))
    counts, bin_edges = np.histogram(baseline_lengths, bins=u_bin_edges)

    prime, unprime = np.meshgrid(counts / len(baseline_lengths), counts / len(baseline_lengths))
    weights = prime * unprime * (2 / (N_antenna - 1)) ** 2
    pyplot.imshow(weights.T, origin='lower')
    pyplot.show()
    return weights


def calculcate_absolute_calibration(sky_averaged_covariance, weights):
    if weights is None:
        absolute_averaged_covariance = sky_averaged_covariance
    else:

        absolute_averaged_covariance = np.zeros_like(sky_averaged_covariance)
        for i in range(sky_averaged_covariance.shape[0]):
            absolute_averaged_covariance[i, ...] = np.mean(sky_averaged_covariance, axis=0)

    return absolute_averaged_covariance


# def uncalibrated_residual_error():
#     raw_variance = np.zeros((len(u), int(len(nu) / 2)))
#
#     for i in range(len(u)):
#         if calibration_type == "redundant":
#             if residuals == "position":
#                 residual_covariance = position_covariance(u[i], 0, nu)
#                 blaah = 0
#             elif residuals == "beam":
#                 residual_covariance = beam_covariance(u[i], v=0, nu=nu, broken_tile_fraction=broken_baselines_weight,
#                                                       calibration_type=calibration_type)
#                 blaah = 0
#             elif residuals == 'both':
#                 residual_covariance = position_covariance(u[i], 0, nu) + \
#                                       beam_covariance(u[i], v=0, nu=nu, broken_tile_fraction=broken_baselines_weight,
#                                                       calibration_type=calibration_type)
#             else:
#                 raise ValueError(f"{residuals} is an invalid residual option for calibration type {calibration_type}")
#
#         elif calibration_type == "sky":
#             if residuals == "sky":
#                 residual_covariance = position_covariance(u[i], 0, nu)
#                 blaah = 0
#             elif residuals == "beam":
#                 residual_covariance = beam_covariance(u[i], v=0, nu=nu, broken_tile_fraction=broken_baselines_weight,
#                                                       calibration_type=calibration_type)
#                 blaah = 0
#             elif residuals == 'both':
#                 residual_covariance = position_covariance(u[i], 0, nu) + \
#                                       beam_covariance(u[i], v=0, nu=nu, broken_tile_fraction=broken_baselines_weight,
#                                                       calibration_type=calibration_type)
#             else:
#                 raise ValueError(f"{residuals} is an invalid residual option for calibration type {calibration_type}")
#
#     raw_variance[i, :] = compute_power(nu, residual_covariance)
#     return raw_variance,


def calibrated_residual_error(u, nu, residuals='both', broken_baselines_weight = 1, weights = None,
                             calibration_type = 'sky', scale_limit = None, N_antenna = 128, tile_diameter=4,
                              position_error = 0.02, model_limit=1):
    cal_variance = np.zeros((len(u), int(len(nu) / 2)))

    if calibration_type == "sky":
        gain_averaged_covariance = gain_error_covariance(u, nu, residuals=residuals, calibration_type=calibration_type,
                                                         weights= weights, N_antenna = N_antenna,
                                                         broken_baseline_weight = broken_baselines_weight,
                                                         tile_diameter=tile_diameter, model_limit=model_limit)

    elif calibration_type == "relative":
        gain_averaged_covariance = gain_error_covariance(u, nu, residuals=residuals, calibration_type="relative",
                                                         weights=weights, N_antenna=N_antenna,
                                                         broken_baseline_weight=broken_baselines_weight,
                                                         tile_diameter=tile_diameter, position_error=position_error)

    elif calibration_type == "absolute":
        sky_averaged_covariance = gain_error_covariance(u, nu, residuals=residuals, calibration_type='sky',
                                                        weights=weights, N_antenna=N_antenna,
                                                        broken_baseline_weight=broken_baselines_weight,
                                                        tile_diameter=tile_diameter, model_limit=model_limit)

        gain_averaged_covariance = calculcate_absolute_calibration(sky_averaged_covariance, weights)

    elif calibration_type == "redundant":
        relative_averaged_covariance = gain_error_covariance(u, nu, residuals=residuals, calibration_type="relative",
                                                             weights=weights, N_antenna = N_antenna,
                                                             broken_baseline_weight = broken_baselines_weight,
                                                             tile_diameter=tile_diameter, position_error=position_error)

        sky_averaged_covariance = gain_error_covariance(u, nu, residuals='both', calibration_type='sky',
                                                        weights=weights, N_antenna=N_antenna,
                                                        broken_baseline_weight=broken_baselines_weight,
                                                        tile_diameter=tile_diameter, model_limit=model_limit)

        absolute_averaged_covariance = calculcate_absolute_calibration(sky_averaged_covariance, weights)

        gain_averaged_covariance = absolute_averaged_covariance + relative_averaged_covariance


    for i in range(len(u)):
        model_covariance = sky_covariance(u[i], 0, nu, S_low=model_limit, S_high=10, tile_diameter=tile_diameter)
        sky_residuals = sky_covariance(u[i], 0, nu, S_low=1e-3, S_high=model_limit, tile_diameter=tile_diameter)
        scale = np.diag(np.zeros_like(nu)) + 1

        if weights is None:
            nu_cov = 2*gain_averaged_covariance*model_covariance + \
                     (scale + 2*gain_averaged_covariance)*sky_residuals
        else:
            nu_cov = 2*gain_averaged_covariance[i, ...]*model_covariance + \
                     (scale + 2*gain_averaged_covariance[i, ...])*sky_residuals

        cal_variance[i, :] = compute_power(nu, nu_cov)

    return cal_variance

def circle_otf(x, a):
    #Fourier Transform of a an Airy Pattern Cubed is also known as the Optical Transfer Function for a Circular
    #Aperture.
    f = np.zeros_like(x)
    index = np.where(np.abs(x) < a)
    x = np.abs(x[index])
    f[index] = a**2./2.*np.arccos(x/a) - a**2./2.*(x/a)*(1.-(x/a)**2.)**(1./2.)
    return f
