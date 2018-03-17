"""
Extract local maxima from a spm, return a csv file with variables:
- x coordinate
- y coordinate
- z coordinate
- peak height
"""

import numpy as np
import pandas as pd


def PeakTable(spm, thresh, mask):
    """
    Make a new array with an extra row/column/plane around the original array

    Parameters
    ----------
    spm : :obj:`numpy.ndarray`
        Statistic map in array form.
    thresh : :obj:`float`
        Voxel-wise threshold to apply to spm.
    mask : :obj:`numpy.ndarray`
        Boolean brain mask in array form.
    """
    buff = 1
    spm_ext = np.pad(spm, buff, 'constant')
    msk_ext = np.pad(mask, buff, 'constant')
    spm_ext = spm_ext * msk_ext
    shape = spm.shape

    # open peak csv
    labels = ['x', 'y', 'z', 'peak']
    peaks = pd.DataFrame(columns=labels)
    # check for each voxel whether it's a peak, if it is, add to table
    for m in range(buff, shape[0]+buff):
        for n in range(buff, shape[1]+buff):
            for o in range(buff, shape[2]+buff):
                if spm_ext[m, n, o] > thresh:
                    surroundings = spm_ext[m-buff:m+buff+1, n-buff:n+buff+1, o-buff:o+buff+1].copy()
                    surroundings[buff, buff, buff] = 0
                    if spm_ext[m, n, o] > np.max(surroundings):
                        res = pd.DataFrame(data=[[m-buff, n-buff, o-buff, spm_ext[m, n, o]]],
                                           columns=labels)
                        peaks = peaks.append(res)
    peaks.index = range(len(peaks))
    return peaks
