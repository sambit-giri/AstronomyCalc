import numpy as np
from astropy import units as u 

import os, requests, zipfile, pkg_resources, wget

class SPARC_Galaxy_dataset:
    def __init__(self):
        '''
        This class can study the files from the 
        SPARC Galaxy dataset (http://astroweb.cwru.edu/SPARC/).
        '''
        self.package_folder = pkg_resources.resource_filename('AstronomyCalc', 'input_data/')
        self.data_folder = os.path.join(self.package_folder, "Rotmod_LTG")
        if not os.path.exists(self.data_folder):
            self.download_data()

    def read_rotation_curves(self, filename=None, name=None):
        assert filename is not None or name is not None
        # Distance = 13.8 Mpc
        quantities = ['Rad','Vobs', 'errV',	
                      'Vgas',	'Vdisk', 'Vbul', 
                      'SBdisk', 'SBbul']		
        units_ = ['kpc',	'km/s',	'km/s',	
                 'km/s','km/s',	'km/s',	
                 'L/pc^2', 'L/pc^2']
        if name is not None:
            filename = os.path.join(self.data_folder, f"{name}_rotmod.dat")
        rd = np.loadtxt(filename)
        values = {quantities[i]:rd[:,i] for i in range(len(quantities))}
        units = {quantities[i]:units_[i] for i in range(len(quantities))}
        return {'values': values, 'units': units}
    
    def download_data(self):
        url = "http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip"
        target_folder = self.package_folder
        wget.download(url, target_folder)

        zip_file_path = os.path.join(target_folder, "Rotmod_LTG.zip")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_file_target = os.path.join(target_folder, "Rotmod_LTG")
            zip_ref.extractall(zip_file_target)

        os.remove(zip_file_path)
        print("Download and extraction successful.")

        return zip_file_target

