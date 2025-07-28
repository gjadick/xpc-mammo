import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
import chromatix.functional as cx
from chromatix.ops import init_plane_resample


h  = 6.62607015e-34           # Planck constant, J/Hz
c = 299792458.0               # speed of light, m/s
J_eV = 1.602176565e-19        # J per eV conversion
PI = jnp.pi


def gaussian2D(x, y, fwhm, normalize=True):
    """
    Generate a 2D Gaussian kernel.
    x, y : 1D arrays
        Grid coordinates [arbitrary length]
    fwhm : float
        Full-width at half-maximum of the Gaussian (units must match x, y)
    normalize : bool
        If True, normalize the kernel to sum to 1
    """
    sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
    X, Y = jnp.meshgrid(x, y)
    kernel = jnp.exp(-(X**2 + Y**2) / (2 * sigma**2))
    if normalize:
        kernel = kernel / jnp.sum(kernel)
    return kernel


def lorentzian2D(x, y, fwhm, normalize=True):
    """
    Generate a 2D Lorentzian kernel.
    x, y : 1D arrays
        Grid coordinates [arbitrary length]
    fwhm : float
        Full-width at half-max of the Lorentzian (units must match x,y)
    normalize : bool
        If True, normalize the kernel to sum to 1
    """
    gamma = fwhm/2
    X, Y = jnp.meshgrid(x, y)
    kernel = gamma / (2 * PI * (X**2 + Y**2 + gamma**2)**1.5)
    if normalize:
        kernel = kernel / jnp.sum(kernel)
    return kernel


def apply_psf(img, dx, psf='lorentzian', fwhm='pixel', kernel_width=0.2):
    """ 
    Apply a point spread function (PSF) to a 2D image via convolution.

    Parameters
    ----------
    img : 2D array (jnp.ndarray)
        The input image to which the PSF will be applied.
    dx : float
        Pixel size in physical units (e.g., mm or µm).
    psf : {'lorentzian', 'gaussian'}, optional
        The type of PSF to apply. Default is 'lorentzian'.
    fwhm : float or {'pixel', None}, optional
        Full width at half maximum of the PSF, in the same units as dx.
        - If 'pixel', sets FWHM to dx (i.e., 1 pixel wide).
        - If None, no PSF is applied (function returns `img` unchanged).
    kernel_width : float, optional
        Fraction of the image field-of-view to use as the PSF kernel width.
        A smaller value reduces computational cost. Default is 0.2.

    Returns
    -------
    img_nonideal : 2D array (jnp.ndarray)
        The image convolved with the PSF kernel, simulating the effect 
        of limited resolution due to the imaging system.

    Notes
    -----
    - Assumes a square image (`img.shape[0] == img.shape[1]`).
    - The kernel is computed over a reduced field-of-view (`kernel_width * FOV`)
      for computational efficiency.
    - Pads the input image with constant edge values before convolution to 
      avoid edge artifacts.
    """

    # Handle spetial FWHM options
    if fwhm is None:
        return img
    elif fwhm == 'pixel':
        fwhm = dx   

    # Check if PSF format is supported
    psf = psf.lower()
    assert psf in ('lorentzian', 'gaussian')

    # Compute reduced FOV for kernel grid for efficiency
    small_FOV = kernel_width * max(img.shape) * dx
    x = jnp.arange(-small_FOV, small_FOV, dx) + dx

    # Generate the kernel (normalized by default)
    if psf == 'lorentzian':
        kernel = lorentzian2D(x, x, fwhm)
    elif psf == 'gaussian':
        kernel = gaussian2D(x, x, fwhm)

    # Compute padding (half kernel size on each size to account for fillvalue = 0)
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    img_pad = jnp.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode='edge')

    # Apply convolution
    img_nonideal = convolve2d(img_pad, kernel, mode='valid')

    return img_nonideal


def simulate_projection(proj_beta, proj_delta, dx, det_N, det_dx, energy, R, 
                        I0=None, det_psf=None, det_fwhm=5e-6, n_medium=1, N_pad=100, key=jax.random.PRNGKey(3)):
    """
    Simulates a single-energy X-ray phase-contrast imaging (XPCI) projection using propagation-based phase contrast.

    Parameters
    ----------
    proj_beta : ndarray
        2D array representing the line integral of the imaginary part of the refractive index (∫ beta dz) 
        at a given X-ray energy.
    proj_delta : ndarray
        2D array representing the line integral of the real part of the refractive index decrement (∫ delta dz)
        at the same X-ray energy as specified by the `energy` argument.
    dx : float
        Pixel size of the input projections (phantom resolution) in meters.
    det_N : int
        Number of detector pixels along one dimension (assumes a square detector).
    det_dx : float
        Detector pixel size in meters.
    energy : float
        X-ray energy in keV used for the simulation. **Must match the energy used to generate `proj_beta` and `proj_delta`.**
    R : float
        Propagation distance (object-to-detector) in meters.
    I0 : float, optional
        Mean incident photon fluence per pixel (used to apply Poisson noise).
    det_psf : callable, optional
        Point spread function (PSF) model for the detector, applied as a blur to the image.
    det_fwhm : float, optional
        Full-width at half maximum (FWHM) of the detector PSF in meters. Default is 1e-6.
    n_medium : float, optional
        Refractive index of the propagation medium (e.g., air = 1.0). Default is 1.
    N_pad : int, optional
        Padding used in the transfer propagation step. Default is 100.
    key : jax.random.PRNGKey, optional
        PRNG key for generating Poisson noise if `I0` is specified.

    Returns
    -------
    img : ndarray
        Simulated detector intensity image, normalized such that the center pixel equals 1.0 before noise and PSF.

    Notes
    -----
    - The input projections `proj_beta` and `proj_delta` **must be computed at the same energy** as the `energy` parameter.
    - The detector field of view (`det_N * det_dx`) **must be less than or equal to** the phantom field of view (`proj_beta.shape[0] * dx`).
    - If `I0` is provided, Poisson noise is applied to simulate quantum noise.
    - If `det_psf` is provided, a PSF blur is applied to simulate detector resolution.
    - This function assumes square images and detectors.
    """

    assert (proj_beta.shape == proj_delta.shape)
    assert det_N * det_dx <= proj_beta.shape[0] * dx, 'Detector FOV must be <= phantom FOV'

    phantom_fov = proj_beta.shape[0] * dx
    det_shape = (det_N, det_N)
    
    field = cx.plane_wave(
        shape = proj_beta.shape, 
        dx = dx,
        spectrum = get_wavelen(energy),
        spectral_density = 1.0,
    )
    field = field / field.intensity.max()**0.5  # normalize
    cval = field.intensity.max()

    exit_field = cx.thin_sample(field, proj_beta[None, ..., None, None], proj_delta[None, ..., None, None], 1.0)
    det_field = cx.transfer_propagate(exit_field, R, n_medium, N_pad, cval=cval, mode='same')

    det_img = det_field.intensity.squeeze()
    if det_psf is not None:
        det_img = apply_psf(det_img, dx, psf=det_psf, fwhm=det_fwhm, kernel_width=0.1)

    det_resample_func = init_plane_resample(det_shape, (det_dx, det_dx), resampling_method='linear')
    img = det_resample_func(det_img[...,None,None], field.dx.ravel()[:1])[...,0,0]
    img /= img.ravel()[0] 

    if I0 is not None:
        img = jax.random.poisson(key, I0*img, img.shape) / I0
        
    return img    