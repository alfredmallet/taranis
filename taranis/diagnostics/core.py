# equation-agnostic diagnostics machinery.
import jax.numpy as jnp
import numpy as np

from .. import _precision


def _perp_bins(params):
    # (kunit, kmax) of the perpendicular bins: min(nx,ny)//2 fundamental units, extended to
    # the largest DEALIASED k_perp where that lies beyond it. In a square box it never does
    # (the 2/3 cut sits inside nx//2), so the bins are the historical ones; in a rectangular
    # box or grid the retained modes of the finer direction would otherwise fall off the
    # last edge and silently vanish from the spectrum.
    from ..physics.gdi import _perp_grids_np   # numpy, setup-time; diagnostics -> physics
    kunit = min(2*np.pi/params.Lx, 2*np.pi/params.Ly)
    kmax = min(params.nx//2, params.ny//2)*kunit
    _, _, ksq, _, _, perp_dealias = _perp_grids_np(params)
    return kunit, max(kmax, float(np.sqrt(np.max(ksq[perp_dealias]))))


def _binned(kmag, energies, kunit, kmax, bin_factor):
    # radial binning shared by perpspec/parspec: (bin_centers, one spectrum per energy)
    dk = kunit*bin_factor
    bin_edges = jnp.arange(0,kmax+dk,dk,dtype=_precision.ftype)
    specs = tuple(jnp.histogram(kmag,bins=bin_edges,weights=e/dk)[0] for e in energies)
    return ((bin_edges[1:] + bin_edges[:-1]) / 2,) + specs
