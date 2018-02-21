import numpy as np
import numpy.linalg as npl
import scipy.constants as const

eps=const.epsilon_0*1e-10*const.e**-2*1e6*const.physical_constants['Avogadro constant'][0]**-1

def getkvec(kmax,box):
    '''Returns an array with all vectors in k space obeying k_i<kmax'''
    args=[np.arange(-kmax,kmax+1)/box[i] for i in range(len(box))]
    k_vecs=np.array(np.meshgrid(*args)).T.reshape(-1,len(box))
    a=(2*np.pi*k_vecs)#[np.sqrt(np.sum(k_vecs**2,axis=1)) <=kmax ]
    return a[np.sum(a**2,axis=1) != 0]


def longrange(pos,q,box,k_max,alpha,potential=True,forces=True):
    '''
    Reciprocal space component of Ewald summation.
    
    Arguments:
    
        pos     (ndarray):      A (n,d)-dimensional numpy-array (d dimensional postions of n particles)
        q       (ndarray):      A n-dimensional numpy-array (charges)      
        box     (ndarray):      A d-dimensional numpy-array (size of preriodic box)
        k_max   (int):          A positive integer (reciprocal space cutoff)
        alpha   (float):        A positive float (constant for division of comutation in real/reciproc space)
        e_0     (float):        A positive float (dielectric constant)
     
    Returns:
        U       (float):        A positive float (total energy of the charge distribution)
        F       (ndarray):      A (n,d)-dimensional numpy-array (force acting on each particle)
        
            ''' 
    
    if not np.sum(q)==0:
        raise ValueError('Total charge needs to be 0, but is '+str(np.sum(q)))
    
    if box.shape[-1] != pos.shape[-1]:
        raise ValueError('Dimension missmatch: postions %iD, box %iD'%(pos.shape[-1],box.shape[-1]))
    
    na=np.newaxis 

    k     = getkvec(k_max,box)
    k2    = npl.norm(k,axis=1)**2
    pre   = 2*np.pi/np.prod(box)
    tmp   = 1/k2*np.exp(- k2 /(alpha**2 * 4))    
    sk    = np.sum(q[:,na]*np.exp(-1j*np.einsum('ki,ji',k,pos)),axis=0)
    
    if potential:
        U = 1/(4*np.pi*eps)*pre*np.sum(tmp*np.real(sk*sk.conj()))
        if not forces:
            return U
    
    if forces:
        F = 1/(4*np.pi*eps*1e-10)*2*pre*np.sum((tmp[na,:,na]*((q[:,na]*np.imag\
              (sk*np.exp(1j*(np.einsum('ki,ji',k,pos)))))\
                [:,:,na]*k[na,:,:])),axis=1)
        if not potential:
            return F
    
    return U, F

def self_energy(q,alpha):
    '''
    Returns the self energy of a charge distribution .
    
    Arguments:
    
        pos     (ndarray):      A (n,d)-dimensional numpy-array (d dimensional postions of n particles)
        q       (ndarray):      A n-dimensional numpy-array (charges)      
        sigma   (float):        A positive float (width of gaussian distribution)
        e_0     (float):        A positive float (dielectric constant)
     
    Returns:
        E       (float):        A positive float (self energy of the charge distribution)
        
            ''' 
    return -1/(4*np.pi*eps)*alpha/np.sqrt(np.pi)*np.sum(q**2)
