import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from inputs.xscatter import get_delta_beta_mix

class Material:
    """
    Represents a material with its physical properties for imaging simulations.

    This class organizes material properties including density and the complex 
    refractive index components (delta and beta) over a range of X-ray energies. 
    These are computed based on the material's composition using 
    `get_delta_beta_mix`.

    Attributes
    ----------
    name : str
        Name of the material.
    matcomp : str
        Specification of the material composition, compatible with get_delta_beta_mix.
        E.g. -- water is 'H(11.2)O(88.2)'
    density : float
        Mass density of the material (g/cm³).
    energy_range : ndarray
        Array of energies (in keV) over which delta and beta are precomputed.
    delta_range : ndarray
        Precomputed real part of the refractive index decrement over energy_range.
    beta_range : ndarray
        Precomputed imaginary part of the refractive index decrement over energy_range.
    """
    def __init__(self, name, matcomp, density):
        self.name = name
        self.matcomp = matcomp
        self.density = density
        self.energy_range = np.linspace(1, 150, 1000)
        self.delta_range, self.beta_range = get_delta_beta_mix(matcomp, self.energy_range, density)

    def db(self, energy):
        """
        Interpolate and return delta and beta at given energy values.

        Parameters
        ----------
        energy : float or ndarray
            Energy value(s) in keV at which to evaluate delta and beta.

        Returns
        -------
        delta : float or ndarray
            Interpolated real part of the refractive index decrement at the input energy.
        beta : float or ndarray
            Interpolated imaginary part of the refractive index decrement at the input energy.
        """
        delta = np.interp(energy, self.energy_range, self.delta_range)
        beta = np.interp(energy, self.energy_range, self.beta_range)
        return delta, beta


def gaussian2d(N, mu=0, sigma=1, normalize=True):
    """
    Creates a 2D Gaussian kernel over a square grid of size N×N, centered at the origin.
    Useful for image filtering, convolution, or simulating point spread functions.

    Parameters:
    ----------
    N : int
        Size of the output kernel (N x N).
    mu : float, optional
        Mean of the Gaussian distribution. Currently unused, as the kernel is always centered at 0 (default is 0).
    sigma : float, optional
        Standard deviation (spread) of the Gaussian (default is 1).
    normalize : bool, optional
        If True, normalize the kernel so that the sum of all elements equals 1 (default is True).

    Returns:
    -------
    G : ndarray
        A 2D NumPy array of shape (N, N) representing the Gaussian kernel.
    """
    x = np.linspace(-N/2, N/2, N)
    X, Y = np.meshgrid(x, x)
    G = 1/(2*np.pi*sigma**2) * np.exp(-(X**2 + Y**2)/(2*sigma**2))
    if normalize:
        G /= G.sum()
    return G


def make_phantom(N, dx, kappa=1, alpha=3.4, f0=50, normalize=True):
    """
    This function creates a volumetric phantom whose spatial frequency content 
    follows a power-law distribution, commonly used to simulate the noise-like 
    texture of anatomical backgrounds in medical imaging (e.g. breast)

    Parameters:
    ----------
    N : int
        Size of the cubic volume (i.e., the output volume will be of shape (N, N, N)).
    dx : float
        Spatial sampling interval (voxel size) in each dimension.
    kappa : float, optional
        Amplitude scaling factor for the power-law spectrum (default is 1).
    alpha : float, optional
        Exponent of the power-law spectrum (default is 3.4, typical for breast tissue).
    f0 : float, optional
        Frequency offset added to avoid singularity at zero frequency (default is 50).
    normalize : bool, optional
        If True, normalize the output volume to have a maximum value of 1 (default is True).

    Returns:
    -------
    vol : ndarray
        A 3D NumPy array of shape (N, N, N) representing the synthetic power-law phantom.
    """
    # Generate the spatial and frequency coordinates.
    x = dx*np.linspace(-N/2, N/2, N)
    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) + f0
    FX, FY, FZ = np.meshgrid(fx, fx, fx)
    F = np.sqrt(FX**2 + FY**2 + FZ**2) 

    # Generate the volume.
    phi = np.pi * (2*np.random.random_sample((N,N,N))-1)
    V = np.exp(-1j*phi) * np.sqrt(kappa / (F+f0)**alpha)
    vol = np.abs(np.fft.ifftn(V))
    
    if normalize:
        vol /= vol.max()
    
    return vol



def tmap_ellipsoid(N, rx, ry, xc=0, yc=0, angle=None):
    """
    Generate a 2D elliptical envelope mask for phantom structuring.

    This function creates an ellipsoidal transparency map (tmap) over a square grid.
    The map defines a soft mask where values inside the ellipse are nonzero and values
    outside are zero, allowing you to spatially modulate background textures (e.g., 
    power-law phantoms) into more anatomically plausible shapes. The ellipse can be
    optionally rotated.

    Parameters
    ----------
    N : int
        Size of the square output array (N x N).
    rx : float
        Semi-axis length of the ellipse in the x-direction.
    ry : float
        Semi-axis length of the ellipse in the y-direction.
    xc : float, optional
        x-coordinate of the ellipse center (default is 0, centered).
    yc : float, optional
        y-coordinate of the ellipse center (default is 0, centered).
    angle : float, optional
        Angle in degrees to rotate the ellipse counterclockwise (default is None, no rotation).

    Returns
    -------
    tmap : ndarray
        A 2D NumPy array of shape (N, N) containing values in [0, 1],
        representing the elliptical mask. Values are 0 outside the ellipse,
        and decrease quadratically toward the boundary inside.
    """
    x = np.linspace(-N/2, N/2, N)
    X, Y = np.meshgrid(x, x)
    tmap = 1 - ((X - xc)/rx)**2 - ((Y - yc)/ry)**2
    tmap[tmap < 0] = 0  
    if angle is not None:
        tmap = rotate(tmap, angle, reshape=False)
    return tmap


def thresh_texture(vol, d):
    """
    Convert a continuous phantom texture into a binary mask using a top percentile threshold.
    This function thresholds the input volume such that a fraction `d` of voxels with the 
    highest intensities are set to 1 (representing one material), and all others are set to 0 
    (representing a second material). It effectively binarizes a background texture into 
    two discrete material regions.

    Note that this might take a while to run depending on your machine and the volume size.

    Parameters
    ----------
    vol : ndarray
        Input 2D or 3D NumPy array containing the background texture (e.g., power-law phantom).
    d : float
        Desired fraction of voxels to assign to the higher-intensity material. 
        Must be between 0 and 1.

    Returns
    -------
    mask : ndarray
        Binary NumPy array of the same shape as `vol`, where the top `d` fraction of intensities 
        are 1 and the rest are 0.
    """

    assert 0 < d < 1, 'd must be between 0 and 1 (exclusive)'
    thresh = np.quantile(vol, 1 - d)
    mask = (vol > thresh).astype(np.uint8)
    return mask


def make_db_vol(vol, mat_dict, energy):
    """
    Generate 3D volumes of delta and beta values for a phantom at given energy or energies.
    For each material label in the phantom volume, this function assigns the corresponding 
    refractive index decrement (delta) and absorption index (beta) values at the specified 
    energy/energies to all voxels of that material. The output volumes represent the spatial 
    distribution of delta and beta throughout the phantom.

    Parameters
    ----------
    vol : ndarray, shape (Nz, Ny, Nx)
        3D NumPy array representing the phantom, where each voxel's value is an integer label 
        identifying the material at that location.
    mat_dict : dict
        Dictionary mapping integer material labels to Material objects. Each Material object 
        must have a `.db(energy)` method that returns (delta, beta) at the given energy.
        There should be a key for each unique value in `vol`, otherwise the default is vacuum.
    energy : float or ndarray of shape (Ne,)
        Energy value(s) at which to compute delta and beta.

    Returns
    -------
    delta_vol : ndarray, shape (Ne, Nz, Ny, Nx)
        3D NumPy array containing the delta values assigned per energy and voxel.
    beta_vol : ndarray, shape (Ne, Nz, Ny, Nx)
        3D NumPy array containing the beta values assigned per energy and voxel.
    """
    energy = np.atleast_1d(energy)
    Ne = energy.size
    
    # Prepare lookup tables for indexing
    max_label = int(vol.max())
    delta_lookup = np.zeros((Ne, max_label+1), dtype=np.float32)
    beta_lookup = np.zeros((Ne, max_label+1), dtype=np.float32)
    for m, mat in mat_dict.items():
        delta, beta = mat.db(energy)
        delta_lookup[:, m] = delta
        beta_lookup[:, m] = beta

    # Create volumes with vectorized indexing
    vol_delta = delta_lookup[:, vol] 
    vol_beta = beta_lookup[:, vol]

    return vol_delta, vol_beta





    