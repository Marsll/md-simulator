
# coding: utf-8

# In[1]:


import numpy as np

class Solver():
    
    def __init__(self,pos,dt,m,f,params):
        self.po    = pos
        self.p     = None
        self.dt    = dt
        self.m     = m
        self.f     = f
        self.params     = params
        
    def newton(self,vn=None):
        if vn is None:
            vn = np.zeros_like(self.po)
        vnn = vn+F(self.po,vn,self.params)/self.m[:,np.newaxis]*self.dt
        self.p = (self.po +(vnn+vn)/2*self.dt)%(self.params[1]-1)

    def verlet(self):
        tmp = self.p
        self.p = (F(self.p,(self.p-self.po)/self.dt,self.params)/self.m[:,np.newaxis]*self.dt**2+2*self.p-self.po)%(self.params[1]-1)
        self.po = tmp
        
    def run(self,steps):
        P=[self.po]
        self.newton()
        for i in range(steps):
            self.verlet()
            P.append(self.po)
        return P
            
    def set_params(self,params):
        print('changed from:  \n'+ str(self.params) +'\n to: \n' +str(params))
        self.params=params
            

