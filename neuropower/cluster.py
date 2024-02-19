"""Extract local maxima from a spm.

Return a csv file with variables:

- x-axis array index (i)
- y-axis array index (j)
- z-axis array index (k)
- peak z-value
- peak p-value
"""

import itertools

import numpy as np
import pandas as pd


def PeakTable(spm, exc, mask):
    """Identify local maxima above z-value threshold in masked statistical \
       image in array form.

    Parameters
    ----------
    spm : :obj:`numpy.ndarray`
        Z-statistic map in array form.
    exc : :obj:`float`
        Voxel-wise z-value threshold (i.e., excursion threshold or cluster-
        defining threshold) to apply to ``spm``.
    mask : :obj:`numpy.ndarray`
        Boolean mask in array form.

    Returns
    -------
    peak_df : :obj:`pandas.DataFrame`
        DataFrame with local maxima (peaks) from statistical map. Each peak is
        provided with i, j, and k indices, z-value, and peak-level p-value.
    """
    r = 1  # radius of cube in voxels
    spm_ext = np.pad(spm, r, "constant")
    msk_ext = np.pad(mask, r, "constant")
    spm_ext = spm_ext * msk_ext
    shape = spm.shape

    # create empty dataframe
    labels = ["i", "j", "k", "zval"]
    peak_df = pd.DataFrame(columns=labels)

    # check for each voxel whether it's a peak. if it is, add to table
    for m, n, o in itertools.product(
        range(r, shape[0] + r), range(r, shape[1] + r), range(r, shape[2] + r)
    ):
        if spm_ext[m, n, o] > exc:
            surroundings = spm_ext[
                m - r : m + r + 1, n - r : n + r + 1, o - r : o + r + 1
            ].copy()
            surroundings[r, r, r] = 0
            if spm_ext[m, n, o] > np.max(surroundings):
                res = pd.DataFrame(
                    data=[[m - r, n - r, o - r, spm_ext[m, n, o]]], columns=labels
                )
                peak_df = peak_df.append(res)

    # Peak-level p-values (not the same as simple z-to-p conversion)
    p_values = np.exp(-float(exc) * (np.array(peak_df["zval"]) - float(exc)))
    p_values[p_values < 10**-6] = 10**-6
    peak_df["pval"] = p_values
    peak_df = peak_df.sort_values(by=["zval"], ascending=False)
    peak_df.index = range(len(peak_df))
    return peak_df
