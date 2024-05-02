import numpy as np
from astropy import units as u 

class SPARC_Galaxy_dataset:
    def __init__(self):
        '''
        This class can study the files from the 
        SPARC Galaxy dataset (http://astroweb.cwru.edu/SPARC/).
        '''
        pass 

    def read_rotation_curves(self, filename):
        # Distance = 13.8 Mpc
        quantities = ['Rad','Vobs', 'errV',	
                      'Vgas',	'Vdisk', 'Vbul', 
                      'SBdisk', 'SBbul']		
        units_ = ['kpc',	'km/s',	'km/s',	
                 'km/s','km/s',	'km/s',	
                 'L/pc^2', 'L/pc^2']
        rd = np.loadtxt(filename)
        values = {quantities[i]:rd[:,i] for i in range(len(quantities))}
        units = {quantities[i]:units_[i] for i in range(len(quantities))}
        return {'values': values, 'units': units}
        