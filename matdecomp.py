# -*- coding: utf-8 -*-
"""
MIT License

Copyright 2019 Florian Schaff <florian.schaff@monash.edu>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

------

Code originally written and tested with:
numpy 1.14.3
Python 3.6


How to use:
    Call spbi_material_basis to perform phase retrieval
    Call adjust_delta to manage rare insabilities with certain delta/beta values
"""

import numpy as np


def spbi_material_basis(images, distances, px, ds, mus):
    """Spectral phase retrieval using a material basis to model the sample.
    The linearised TIE without crossterm is inverted directly. The algorithm
    returns the projected thicknesses for both materials. If more than two
    images are provided, the least-squares solution is calculated.

    - Any combination of materials and images at any energy/distance
      combinations are possible.
    - Generally, any third material can be expressed as a linear
      combination of the first two materials unless there are k-edge
      involved.
    - Multiple materials typically requires a change in energy for a stable
      algorithm. The number of different measurements should be the same or
      more than the number of materials.

    florian.schaff@monash.edu

    Parameters
    ----------
    images : list of 2-d arrays / 3-d array
        input images containing the measurements without log taken. Can be any
        number of images >= 2
    distances : float or list of floats
        propagation distances [m]. Can either be a single float if the same
        propagation distance is used for all measurements, or a list of same
        length as the number of images
    px : float
        detector pixel size of the measurements
    ds, mus : 2-d array
        delta and mu values to be used. Different energies along the first
        axis, different materials along the second axis.

    Returns
    ----------
    out : TR
        phase retrieved thicknesses for the different materials

    """

    assert ds.shape == mus.shape
    assert len(images) == ds.shape[0]

    distances = np.atleast_1d(distances)
    if distances.shape[0] == 1:
        distances = np.tile(distances, len(images))

    assert len(distances) == len(images)

    iA = _spbi_mb_solver_matrix(images[0].shape, distances, px, ds, mus)
    b = np.fft.fftn(np.moveaxis(-np.log(images), 0, 2), axes=(0, 1))
    x = np.einsum('ijxy, xyj -> xyi', iA, b)
    return np.real(np.moveaxis(np.fft.ifftn(x, axes=(0, 1)), 2, 0))


def adjust_delta(ds, mus, ens):
    """Slightly adjust delta values in the event of constant and quartic terms
    with opposite signs in the determinant during phase retrieval when using
    two mats and two energies.

    Note: The quadratic term could still have an opposite sign, in which case
    a third measurement is necessary.

    florian.schaff@monash.edu

    Parameters
    ----------
    ens : list of float
        x-ray energies of the measurements [keV]. Must be of length 2
    ds, mus : array of shape (2, 2)
        delta and mu values. Values for different energies along the first
        axis, different materials along the second axis.

    Returns
    ----------
    out : array of shape (2, 2)
        ds with scaling applied if necessary

    """
    # if np.prod(np.array(_test_sign(ds, mus, True))[[0, -1]]) == -1:
    if np.prod(np.array(_test_sign(ds, mus))[[0, -1]]) == -1:
        print('Delta and mu term signs don\'t match, applying scaling eq.')
        return np.outer(ens[0]**2/np.array(ens)**2, ds[0])
    else:
        return ds


def _test_sign(ds, mus):
    """test for the sign of the denominator terms in 2mat/2energy sPBI

    florian.schaff@monash.edu

    """
    const_term = mus[1, 1] * mus[0, 0] - mus[1, 0] * mus[0, 1]
    k2_term = mus[1, 1] * ds[0, 0] + mus[0, 0] * ds[1, 1] \
              - mus[1, 0] * ds[0, 1] - mus[0, 1] * ds[1, 0]
    k4_term = ds[1, 1] * ds[0, 0] - ds[1, 0] * ds[0, 1]
    return (np.sign(const_term), np.sign(k2_term), np.sign(k4_term))


def _inv_array_matrix(M):
    """Function to be used with _spbi_mb_solver_matrix. Inverts prod(sh) (i,j)
    matrices. Index order needs to be (sh, i, j). Returns an (i, j, sh) array.

    florian.schaff@monash.edu

    """
    if M.shape[-2:] == (2, 2):
        A, B, C, D = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
        iA = 1./A
        iDCAB = 1./(D - C * iA * B)
        return np.array([[iA + iA * B * iDCAB * C * iA, -iA * B * iDCAB],
                         [-iDCAB * C * iA, iDCAB]])
    else:
        return np.moveaxis(np.linalg.inv(M), [-2, -1], [0, 1]).copy()


def _spbi_mb_solver_matrix(sh, distances, px, ds, mus):
    """Create solution matrix for spectral phase retrieval using a material
    basis to model the sample.

    florian.schaff@monash.edu

    Parameters
    ----------
    sh : tuple
        shape of the individual input images, e.g. (2048, 2048)
    distances : 1-d array
        propagation distances for each image in [m]. Length must match first
        axis of ds/mus
    px : float
        detector pixel size of the measurements
    ds, mus : 2-d array
        delta and mu values. Values for different energies along the first
        axis, different materials along the second axis.

    Returns
    ----------
    out : 4-d array
        (i, j, x, y) matrix iA from which the phase retrieval solution can be
        calculated for a (j, x, y) stack of input images b as e.g.
        xF = numpy.einsum('ijxy, jxy-> ixy', iA, b)
        x = numpy.real(np.fft.ifftn(xF, axes=(-1, -2)))
    """
    # This code is optimized for speed in numpy 1.14.3. The speed depends
    # a lot on the order of the involved arrays ('C'/'F') and the einsum
    # indices are optimized for the normal use here.  moveaxis and .copy() also
    # increase the speed by a large factor.
    K2 = np.multiply.outer(4 * np.pi**2 * distances[:, None] * ds,
                           np.add.outer(np.square(np.fft.fftfreq(sh[0], px)),
                                        np.square(np.fft.fftfreq(sh[1], px))))

    A = np.multiply.outer(mus, np.ones(sh)) + K2
    # for 2 materials and energies solve directly, else use normal equation
    if ds.shape == (2, 2):
        iATA = _inv_array_matrix(np.moveaxis(A, [0, 1], [2, 3]))
    else:
        # print('More than 2 images, calculating LSQ solution')
        ATA = np.einsum('jixy, jkxy -> xyik', A, A)
        iATA = np.einsum('ijxy, kjxy -> ikxy', _inv_array_matrix(ATA), A)
    return iATA.astype(np.complex128)  # complex to speed up @ with FFT images
