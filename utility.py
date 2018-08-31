import h5py
import numpy as np

def load_h5_dataset(file_name):
    h5f = h5py.File(file_name,'r')
    variable = h5f['variable'][:]
    h5f.close()
    return variable

def save_numpy_to_h5_dataset(file_name, variable):
    h5f = h5py.File(file_name + '.h5', 'w')
    h5f.create_dataset('variable', data=variable)
    h5f.close()
    

def single_frame_to_numpy(img, index):
    img_numpy = np.zeros((img.dataobj.shape[0],img.dataobj.shape[1],img.dataobj.shape[2]))
    img_numpy = img.dataobj[:,:,:,index]
    img_numpy = img_numpy / np.max(img_numpy) # normalize
    img_numpy = img_numpy.astype('float32')
    img_numpy = np.expand_dims(img_numpy, axis=0)
    img_numpy = np.expand_dims(img_numpy, axis=0)
    return img_numpy