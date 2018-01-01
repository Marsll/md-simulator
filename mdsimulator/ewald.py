
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt


def getkvec(kmax,box):
    '''Returns an array with all vectors in k space obeying k_i<kmax'''
    args=[np.arange(0,kmax)/box[i] for i in range(len(box))]
    return 2*np.pi*np.array(np.meshgrid(*args)).T.reshape(-1,len(box))

def longrange(pos,q,box,k_max,sigma,e_0):
    '''
    Reciprocal space component of Ewald summation.
    
    Arguments:
    
        pos     (ndarray):      A (n,d)-dimensional numpy-array (d dimensional postions of n particles)
        q       (ndarray):      A n-dimensional numpy-array (charges)      
        box     (ndarray):      A d-dimensional numpy-array (size of preriodic box)
        k_max   (int):          A positive integer (reciprocal space cutoff)
        sigma   (float):        A positive float (width of gaussian distribution)
        e_0     (float):        A positive float (dielectric constant)
     
    Returns:
        U       (float):        A positive float (total energy of the charge distribution)
        F       (ndarray):      A (n,d)-dimensional numpy-array (force acting on each particle)
        
            ''' 
    
    if not np.sum(q)==0:
        raise ValueError('Total charge needs to be 0, but is '+str(np.sum(q)))
    
    if box.shape[-1] != pos.shape[-1]:
        raise ValueError('Dimension missmatch: postions %iD, box %iD'%(pos.shape[-1],box.shape[-1]))
    
    na=np.newaxis #reduces length of code drastically
    
    pre = 1/(np.prod(box)*e_0) 
    
    k   = getkvec(k_max,box)[1:] 
    k2  = npl.norm(k,axis=1)**2

    sk  = np.sum(q[:,na] * np.exp(1j * np.einsum("ki,ji",k,pos)),axis=0)
    sk2 = np.abs(sk)**2
    
    tmp = pre * np.exp(-sigma**2/2 * k2) / k2
    
    U   = np.sum(tmp * sk2)
    F   = -np.sum((2 * tmp[:,na,na] * ((q[:,na] * np.imag(np.exp(-1j * np.einsum("ki,ji",k,pos)) * sk))  
                                       .T[:,na,:] * k[:,:,na])).T,axis=-1)

    return (U,F)

def selfenergy(pos,q,sigma,e_0):
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
    E = np.sum(q**2) / (2  * e_0 * (2 * np.pi)**(3/2) * sigma) 
    return E





