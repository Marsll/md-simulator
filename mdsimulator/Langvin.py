
# coding: utf-8

# In[1]:


def F(pos,vel,params,gamma=1,T=290):
    m      = params[0]
    box    = params[1]
    alpha  = params[2]
    r_cut  = params[3]
    k_max  = params[4]
    q      = params[5]
    P      = params[6]
    gamma  = params[7]
    d=0.5*1e-4
    na=const.physical_constants['Avogadro constant'][0]
    
    nl  = NeighborList(box, pos, r_cut)
    nbs = create_nb_order(box, r_cut)
    lj = forces(pos, P, 1/(np.sqrt(2)*alpha), nl, nbs, r_cut, lj=True, coulomb=False)
    cs = forces(pos, P, 1/(np.sqrt(2)*alpha), nl, nbs, r_cut, lj=False, coulomb=True)
    cl = longrange(pos,q,box,k_max,alpha,potential=False,forces=True)
    fr = -gamma*m[:,np.newaxis]*vel
    hb = np.sqrt(2*const.k*T*gamma*na)*(np.sqrt(m)[:,np.newaxis]*d*(2*np.random.random(pos.shape)-1))
    return cl+cs+lj+hb+fr

