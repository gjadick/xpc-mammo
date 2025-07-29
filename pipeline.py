import numpy as np
from phantom import make_phantom, thresh_texture, make_db_vol, tmap_ellipsoid
from phantom import Material
from simulate import simulate_projection
from inputs.xscatter import get_wavenum
from matdecomp import spbi_material_basis


def compute_fom(gt_bmi1, gt_bmi2, bmis):
    Fgt = gt_bmi1.sum() / (gt_bmi1.sum() + gt_bmi2.sum())
    Fmd = bmis[0].sum() / (bmis[0].sum() + bmis[1].sum())
    fom = 100 * abs(Fmd - Fgt) / Fgt
    return fom, Fgt, Fmd


def run_material_decomposition(energies, R, det_dx, mat_frac=0.5, thickness=0.01, seed=33, I0=1e5):
    gland = Material('gland', 'H(10.2)C(18.4)N(3.2)O(67.6)', 1.04)
    adip = Material('adipose', 'H(11.2)C(61.9)N(1.7)O(25.1)', 0.93)
    mat_dict = {0: gland, 1: adip}

    N = 256
    dx = 2e-6
    Nz = int(thickness / 1e-4)
    dz = thickness / Nz
    det_N = int(dx * N // det_dx)

    np.random.seed(seed)
    struct = np.zeros([N, N])
    struct[tmap_ellipsoid(N, 0.35 * N, 0.42 * N, angle=20) > 1e-3] = 1

    vol = make_phantom(N, dx, alpha=4)[:Nz]
    vol_mask = thresh_texture(vol, mat_frac)

    imgs = []
    for energy in energies:
        vol_delta, vol_beta = make_db_vol(vol_mask, mat_dict, energy)
        proj_delta = dz * struct * np.sum(vol_delta[0], axis=0)
        proj_beta = dz * struct * np.sum(vol_beta[0], axis=0)

        img = simulate_projection(
            proj_beta, proj_delta, dx, det_N, det_dx, energy, R,
            I0=I0, det_psf='gaussian', det_fwhm=5e-6
        )
        imgs.append(img)

    imgs = np.array(imgs)

    ds1, bs1 = gland.db(energies)
    ds2, bs2 = adip.db(energies)
    ds = -np.array([ds1, ds2])
    mus = 2 * np.array([bs1, bs2]) * get_wavenum(energies)

    bmis = spbi_material_basis(imgs, R, det_dx, ds.T, mus.T)
    bmis = bmis.clip(0, None)

    gt_bmi1 = dz * struct * np.sum(vol_mask == 0, axis=0)
    gt_bmi2 = dz * struct * np.sum(vol_mask == 1, axis=0)

    fom, Fgt, Fmd = compute_fom(gt_bmi1, gt_bmi2, bmis)

    return {
        'fom': fom,
        'Fgt': Fgt,
        'Fmd': Fmd,
        'imgs': imgs,
        'bmis': bmis,
        'gt_bmi1': gt_bmi1,
        'gt_bmi2': gt_bmi2
    }