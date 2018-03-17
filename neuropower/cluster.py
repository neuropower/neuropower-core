"""
Extract local maxima from a spm, return a csv file with variables:
- x-axis array index (i)
- y-axis array index (j)
- z-axis array index (k)
- peak z-value
- peak p-value
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def PeakTable(spm, thresh, mask):
    """
    Make a new array with an extra row/column/plane around the original array

    Parameters
    ----------
    spm : :obj:`numpy.ndarray`
        Z-statistic map in array form.
    thresh : :obj:`float`
        Voxel-wise z-value threshold to apply to ``spm``.
    mask : :obj:`numpy.ndarray`
        Boolean mask in array form.
    """
    r = 1  # radius of cube in voxels
    spm_ext = np.pad(spm, r, 'constant')
    msk_ext = np.pad(mask, r, 'constant')
    spm_ext = spm_ext * msk_ext
    shape = spm.shape

    # open peak csv
    labels = ['i', 'j', 'k', 'zval']
    peak_df = pd.DataFrame(columns=labels)
    # check for each voxel whether it's a peak. if it is, add to table
    for m in range(r, shape[0]+r):
        for n in range(r, shape[1]+r):
            for o in range(r, shape[2]+r):
                if spm_ext[m, n, o] > thresh:
                    surroundings = spm_ext[m-r:m+r+1, n-r:n+r+1, o-r:o+r+1].copy()
                    surroundings[r, r, r] = 0
                    if spm_ext[m, n, o] > np.max(surroundings):
                        res = pd.DataFrame(data=[[m-r, n-r, o-r, spm_ext[m, n, o]]],
                                           columns=labels)
                        peak_df = peak_df.append(res)

    # Unadjusted p-values (not used)
    p_values = norm.sf(abs(peak_df['zval']))

    # Adjusted p-values (used)
    p_values = np.exp(-float(thresh)*(np.array(peak_df['zval'])-float(thresh)))

    p_values[p_values < 10**-6] = 10**-6
    peak_df['pval'] = p_values
    peak_df = peak_df.sort_values(by=['zval'], ascending=False)
    peak_df.index = range(len(peak_df))
    return peak_df
