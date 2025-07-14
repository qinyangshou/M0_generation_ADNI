import numpy as np
import h5py
from torch.utils.data import Dataset


class LOFT_M0Data(object):
    
    def __init__(self, h5_data, split_dict, prop_cull=None):
        
        self.data = h5_data
        self.split_dict = split_dict
        self.prop_cull = [0.,0.] if prop_cull is None else prop_cull
        
    def _get_id_slice_lst(self, id_lst):
        
        # M0 should be in the shape of x by y by z(num slices)
        slice_range_lst = [(0,np.array( self.data[x + '/M0/dset_M0']).shape[2]) for x in id_lst]
        # remove the edge slices
        slice_range_lst = [(
            np.floor(self.prop_cull[0] * x[1]).astype("int"), # start index
            np.floor(x[1] - self.prop_cull[1] * x[1]).astype("int"), # end index
        ) for x in slice_range_lst] # cull slices since they may be air
        
        id_slice_lst = [[(x, y) for y in range(*y)] for x, y in zip(id_lst, slice_range_lst)]
        combined_lst = []
        for item in id_slice_lst:
            for sub_item in item:
                combined_lst.append(sub_item)
        return combined_lst
    
    def generate_dataset(self, mode, *args, **kwargs):
        
        assert mode in self.split_dict
        id_lst = self.split_dict[mode]
        id_slice_lst = self._get_id_slice_lst(id_lst)
            
        return LOFT_M0_Dataset(self.data, id_slice_lst, use_mask = False)
    
class LOFT_M0_Dataset(Dataset):
    
    def __init__(self, h5_path, id_slice_lst, use_mask = False):
        # load the data from the hdf5 file
        data_conn = h5py.File(h5_path,"r")
        self.data = data_conn
        self.id_slice_lst = id_slice_lst
        self.use_mask = use_mask
        
    def __len__(self):
        
        return len(self.id_slice_lst)
        
    def __getitem__(self, indx):
        
        current_id, current_slice = self.id_slice_lst[indx]
        input_key = current_id + '/M0/dset_M0'
        input_img = self.data[input_key][:,:,current_slice]
        
        control_img = current_id + 'c'
        # normalize data to [0,1]
        img_min, img_max = np.min(input_img), np.max(input_img)
        input_img = (input_img - img_min) / (img_max - img_min)
        
        # currently just return M0 image
        input_img = np.expand_dims(input_img,axis=0)
        input_dict = {}
        input_dict["image"] = input_img
        #input_dict["condition"] = control_img
        return input_dict
