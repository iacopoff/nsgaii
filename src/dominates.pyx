import numpy as np
cimport numpy as cnp
import cython

cpdef dominates(cnp.ndarray[cnp.double_t,ndim=2] a,
              cnp.ndarray[cnp.double_t,ndim=2] b):
    #assert len(a.shape) > 1 and len(b.shape) > 1
    #pdb.set_trace()
    cdef int l = a.shape[0]
    cdef int r = a.shape[1]

    if l >1:
        ret = (np.sum(a <= b,axis =1) == r) & (np.sum(a < b,axis=1) >0)
    else:
        ret = (np.sum(a <= b) == l) & (np.sum(a < b) >0)
    return ret
