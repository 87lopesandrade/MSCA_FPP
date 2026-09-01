import numpy as np
import scipy.io as sio

def convert_calibration():
    # Load the simplified .mat file exported by MATLAB
    data = sio.loadmat('extracted_params.mat')
    
    # Extract matrices
    mtx_l = data['mtx_l']
    dist_l = data['dist_l']
    
    mtx_r = data['mtx_r']
    dist_r = data['dist_r']
    
    R = data['R'].T  # Transpose R to convert from MATLAB row-major to OpenCV column-major
    T = data['T']
    
    # Save as .npz
    np.savez('stereo_params.npz', 
             mtx_l=mtx_l, 
             dist_l=dist_l, 
             mtx_r=mtx_r, 
             dist_r=dist_r, 
             R=R, 
             T=T)
    print("Conversão concluída com sucesso! Salvo em stereo_params.npz.")

if __name__ == '__main__':
    convert_calibration()
