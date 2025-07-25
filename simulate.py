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




    