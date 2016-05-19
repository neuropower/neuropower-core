import nibabel as nib
import numpy as np
import pandas as pd

"""
Extract local maxima from a spm, return a csv file with variables:
- x coordinate
- y coordinate
- z coordinate
- peak height
"""

def PeakTable(spm,exc,mask):
	# make a new array with an extra row/column/plane around the original array
	spm_newdim = tuple(map(lambda x: x+2,spm.shape))
	spm_ext = np.zeros((spm_newdim))
	msk_ext = np.zeros((spm_newdim))
	spm_ext.fill(-100)
	spm_ext[1:(spm.shape[0]+1),1:(spm.shape[1]+1),1:(spm.shape[2]+1)] = spm
	msk_ext[1:(spm.shape[0]+1),1:(spm.shape[1]+1),1:(spm.shape[2]+1)] = mask
	spm_ext = spm_ext * msk_ext
	shape = spm.shape
	spm = None
	# open peak csv
	labels = ['x','y','z','peak']
	peaks = pd.DataFrame(columns=labels)
	# check for each voxel whether it's a peak, if it is, add to table
	for m in xrange(1,shape[0]+1):
		for n in xrange(1,shape[1]+1):
			for o in xrange(1,shape[2]+1):
				surroundings = None
				res = None
				val = None
				maxval = None
				if spm_ext[m,n,o]>exc:
					surroundings=[spm_ext[m-1,n-1,o-1],
					spm_ext[m-1,n-1,o],
					spm_ext[m-1,n-1,o+1],
					spm_ext[m-1,n,o-1],
					spm_ext[m-1,n,o],
					spm_ext[m-1,n,o+1],
					spm_ext[m-1,n+1,o-1],
					spm_ext[m-1,n+1,o],
					spm_ext[m-1,n+1,o+1],
					spm_ext[m,n-1,o-1],
					spm_ext[m,n-1,o],
					spm_ext[m,n-1,o+1],
					spm_ext[m,n,o-1],
					spm_ext[m,n,o+1],
					spm_ext[m,n+1,o-1],
					spm_ext[m,n+1,o],
					spm_ext[m,n+1,o+1],
					spm_ext[m+1,n-1,o-1],
					spm_ext[m+1,n-1,o],
					spm_ext[m+1,n-1,o+1],
					spm_ext[m+1,n,o-1],
					spm_ext[m+1,n,o],
					spm_ext[m+1,n,o+1],
					spm_ext[m+1,n+1,o-1],
					spm_ext[m+1,n+1,o],
					spm_ext[m+1,n+1,o+1]]
					if spm_ext[m,n,o] > np.max(surroundings):
						res =pd.DataFrame(data=[[m-1,n-1,o-1,spm_ext[m,n,o]]],columns=labels)
						peaks=peaks.append(res)
	peaks = peaks.set_index([range(len(peaks))])
	return peaks
