import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append('/ifs/loni/groups/loft/qinyang/ADNI_M0generation/M0_generation/data/M0_Control')

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
            
        return LOFT_M0_Dataset(self.data, id_slice_lst, *args, **kwargs)
    
class LOFT_M0_Dataset(Dataset):
    
    def __init__(self, h5_data, id_slice_lst, preproc_lst = [], use_mask = False, condition = None, normalize = None, fake_RGB = True):
        
        self.data = h5_data
        self.id_slice_lst = id_slice_lst
        self.use_mask = use_mask
        self.condition = condition
        self.preproc_lst = preproc_lst
        self.normalize = normalize
        self.fake_RGB = fake_RGB

    def __len__(self):
        
        return len(self.id_slice_lst)
        
    def __getitem__(self, indx):
        
        current_id, current_slice = self.id_slice_lst[indx]
        input_key = current_id + '/M0/dset_M0'
        input_img = self.data[input_key][:,:,current_slice]
        
        control_key = current_id + '/control/dset_control'
        cond_img = self.data[control_key][:,:,current_slice]
        
        # normalize data to [0,1]
        if self.normalize == 'NormControl':
            img_min, img_max = np.min(cond_img), np.max(cond_img)
            input_img = (input_img - img_min) / (img_max - img_min)
            cond_img  = (cond_img  - img_min) / (img_max - img_min)
        elif self.normalize == 'Norm4096_0center':
            input_img = input_img / 4096 * 2 - 1
            cond_img  = cond_img  / 4096 * 2 - 1
        # currently just return M0 image
        # input_img = np.expand_dims(input_img,axis=0)
        if self.condition is None:
            input_img, _ = self._process_data(input_img, input_img)             
            return input_img
        else: 
            input_img, cond_img = self._process_data(input_img, cond_img)

            if self.fake_RGB:
                input_img = np.expand_dims(input_img,axis = -1)            
                cond_img  = np.expand_dims(cond_img, axis = -1)
                input_img = np.concatenate((input_img,input_img,input_img),axis = -1)
                cond_img  = np.concatenate((cond_img, cond_img, cond_img), axis = -1)
                return input_img, cond_img
            else:
                input_img = np.expand_dims(input_img,axis = 0)            
                cond_img  = np.expand_dims(cond_img, axis = 0) 
                return cond_img, input_img
    
    def _process_data(self, input_img, cond_img):
            
        input_img, cond_img = self._apply_preproc(input_img, cond_img)
        

        # convert to float32
        input_img = input_img.astype("float32")
        cond_img = cond_img.astype("float32")

        # return data
        return input_img, cond_img
    
    def _apply_preproc(self, input_img, cond_img):

        for curr_preproc in self.preproc_lst:            
            input_img, cond_img = curr_preproc(input_img, cond_img)

        return input_img, cond_img

    def get_slice_indicies(self, current_sub): ##sqy used for prediction
        # make into df and caluclate max index
        # make into df
        df = pd.DataFrame(self.id_slice_lst, columns=["key", "slice_indx"])
        df = df[df["key"] == current_sub]

        split_indices = [df.index.tolist()]

        return split_indices

## sqy added to try direct fitting CBF
class LOFT_CBF_Dataset(Dataset):
    
    def __init__(self, h5_data, id_slice_lst, preproc_lst = [], use_mask = False, condition = None, normalize = None, fake_RGB = True):
        
        self.data = h5_data
        self.id_slice_lst = id_slice_lst
        self.use_mask = use_mask
        self.condition = condition
        self.preproc_lst = preproc_lst
        self.normalize = normalize
        self.fake_RGB = fake_RGB

    def __len__(self):
        
        return len(self.id_slice_lst)
        
    def __getitem__(self, indx):
        
        current_id, current_slice = self.id_slice_lst[indx]
        input_key = current_id + '/M0/dset_M0'
        input_img = self.data[input_key][:,:,current_slice]
        
        control_key = current_id + '/control/dset_control'
        control_img = self.data[control_key][:,:,current_slice]

        perfusion_key = current_id + '/perfusion/dset_perf'
        perf_img = self.data[perfusion_key][:,:,current_slice]

        CBF_key = current_id + '/CBF/dset_CBF'
        CBF_img = self.data[CBF_key][:,:,current_slice]
        # normalize data to [0,1]
        if self.normalize == 'NormControl':
            img_min, img_max = np.min(control_img), np.max(control_img)
            input_img = (input_img - img_min) / (img_max - img_min)
            control_img  = (control_img  - img_min) / (img_max - img_min)
        elif self.normalize == 'Norm_CBF200_Control_4096':
            CBF_img = CBF_img / 200 * 2 - 1             # previously constrained to be 0-200
            perf_img = perf_img / 50 * 2 - 1  # roughly scaled to [-1,1] with clip, may need to tune this parameter
            perf_img = np.clip(perf_img, -1, 1)
            control_img  = control_img  / 4096 * 2 - 1
        else:
            print("Not implemented normalization, the results will be problematic")

        if self.condition is None:
            CBF_img,_,_ = self._process_data([CBF_img, CBF_img, CBF_img])             
            return input_img
        else: 
            CBF_img, perf_img, control_img = self._process_data([CBF_img, perf_img,control_img])

            if self.fake_RGB:
                CBF_img = np.expand_dims(CBF_img,axis = -1)            
                perf_img  = np.expand_dims(perf_img, axis = -1)
                control_img  = np.expand_dims(control_img, axis = -1)
                CBF_img = np.concatenate((CBF_img,CBF_img,CBF_img),axis = -1)
                perf_img  = np.concatenate((perf_img, perf_img, perf_img), axis = -1)
                control_img  = np.concatenate((control_img, control_img, control_img), axis = -1)
                return CBF_img, [perf_img, control_img]
            else:
                CBF_img = np.expand_dims(CBF_img,axis = 0)  
                perf_img = np.expand_dims(perf_img,axis = 0)          
                control_img  = np.expand_dims(control_img, axis = 0) 
                return CBF_img, [perf_img, control_img]
    
    def _process_data(self, img_list):
            
        img_list_processed = self._apply_preproc(img_list)
        
        # convert to float32
        img_list_processed = [img.astype("float32") for img in img_list]

        # return data, the first should be the target image, the others should be the conditions
        return img_list_processed
    
    def _apply_preproc(self, img_list):

        for curr_preproc in self.preproc_lst:            
            img_list = curr_preproc(img_list)

        return img_list

    def get_slice_indicies(self, current_sub): ##sqy used for prediction
        # make into df and caluclate max index
        # make into df
        df = pd.DataFrame(self.id_slice_lst, columns=["key", "slice_indx"])
        df = df[df["key"] == current_sub]

        split_indices = [df.index.tolist()]

        return split_indices
       
import h5py
import pandas as pd
from image_processing import AugmentRandomRotation, AugmentRandomRotationMultiple, AugmentHorizonalFlip, AugmentHorizonalFlipMultiple
############################################################################################################################################
# M0 datasets
class LOFT_M0_Dataset_Train(LOFT_M0_Dataset):

    def __init__(self, AUG_ROT_RANGE = (-60, 60),  **kwargs):
        h5_path = '/ifs/loni/groups/loft/qinyang/M0_generation_data/data_hdf5_csv/M0_generation_ADNI_030524_alldata.hdf5'
        data_conn = h5py.File(h5_path,"r")
        csv_path = '/ifs/loni/groups/loft/qinyang/M0_generation_data/data_hdf5_csv/M0_generation_ADNI_030524_alldata.csv'
        csv = pd.read_csv(csv_path)
        data_splits = {
            "train": csv[csv["train_val_test"] == "TRAIN"]["subject_ID"].tolist(),
        }
        M0_data = LOFT_M0Data(data_conn, data_splits)        
        train_preproc_lst = [
            AugmentRandomRotation(AUG_ROT_RANGE),
            AugmentHorizonalFlip(),        
        ]        
        id_lst = data_splits["train"]
        id_slice_lst = M0_data._get_id_slice_lst(id_lst)
        super().__init__(data_conn, id_slice_lst, preproc_lst=train_preproc_lst, **kwargs)
    
class LOFT_M0_Dataset_Valid(LOFT_M0_Dataset):

    def __init__(self,  **kwargs):
        h5_path = '/ifs/loni/groups/loft/qinyang/M0_generation_data/data_hdf5_csv/M0_generation_ADNI_030524_alldata.hdf5'
        data_conn = h5py.File(h5_path,"r")
        csv_path = '/ifs/loni/groups/loft/qinyang/M0_generation_data/data_hdf5_csv/M0_generation_ADNI_030524_alldata.csv'
        csv = pd.read_csv(csv_path)
        data_splits = {
            "train": csv[csv["train_val_test"] == "TRAIN"]["subject_ID"].tolist(),
            "valid": csv[csv["train_val_test"] == "VALID"]["subject_ID"].tolist(),
        }
        M0_data = LOFT_M0Data(data_conn, data_splits) 
        id_lst = data_splits["valid"]
        id_slice_lst = M0_data._get_id_slice_lst(id_lst)       
        super().__init__(data_conn, id_slice_lst,  **kwargs)

class LOFT_M0_Dataset_Train_Mayo(LOFT_M0_Dataset):

    def __init__(self, AUG_ROT_RANGE = (-60, 60),  **kwargs):
        h5_path = '/ifs/loni/groups/loft/qinyang/M0_generation_data/data_hdf5_csv/M0_generation_Mayo_012225.hdf5'
        data_conn = h5py.File(h5_path,"r")
        csv_path = '/ifs/loni/groups/loft/qinyang/M0_generation_data/data_hdf5_csv/M0_generation_Mayo_012225.csv'
        csv = pd.read_csv(csv_path)
        data_splits = {
            "train": csv[csv["train_val_test"] == "TRAIN"]["subject_ID"].tolist(),
        }
        M0_data = LOFT_M0Data(data_conn, data_splits)        
        train_preproc_lst = [
            AugmentRandomRotation(AUG_ROT_RANGE),
            AugmentHorizonalFlip(),        
        ]        
        id_lst = data_splits["train"]
        id_slice_lst = M0_data._get_id_slice_lst(id_lst)
        super().__init__(data_conn, id_slice_lst, preproc_lst=train_preproc_lst, **kwargs)
    
class LOFT_M0_Dataset_Valid_Mayo(LOFT_M0_Dataset):

    def __init__(self,  **kwargs):
        h5_path = '/ifs/loni/groups/loft/qinyang/M0_generation_data/data_hdf5_csv/M0_generation_Mayo_012225.hdf5'
        data_conn = h5py.File(h5_path,"r")
        csv_path = '/ifs/loni/groups/loft/qinyang/M0_generation_data/data_hdf5_csv/M0_generation_Mayo_012225.csv'
        csv = pd.read_csv(csv_path)
        data_splits = {
            "train": csv[csv["train_val_test"] == "TRAIN"]["subject_ID"].tolist(),
            "valid": csv[csv["train_val_test"] == "VALID"]["subject_ID"].tolist(),
        }
        M0_data = LOFT_M0Data(data_conn, data_splits) 
        id_lst = data_splits["valid"]
        id_slice_lst = M0_data._get_id_slice_lst(id_lst)       
        super().__init__(data_conn, id_slice_lst,  **kwargs)

from torch.utils.data import ConcatDataset, DataLoader

# Assuming old_dataset and new_dataset are PyTorch datasets
class Combined_M0_Dataset(LOFT_M0_Dataset):
    
    def __init__(self, old_dataset, new_dataset, ratio_new = 0.5):
        self.old_dataset = old_dataset
        self.new_dataset = new_dataset
        self.ratio_new = ratio_new  # Control the proportion, not used for now

    def __len__(self):
        return len(self.old_dataset) + len(self.new_dataset)  # Approximate size

    def __getitem__(self, idx):
        # if random.random() < self.ratio_new:
        #     return self.new_dataset[random.randint(0, len(self.new_dataset) - 1)]
        # else:
        #     return self.old_dataset[random.randint(0, len(self.old_dataset) - 1)]
        if idx < len(self.old_dataset):
            return self.old_dataset[idx]
        else:
            return self.new_dataset[idx - len(self.old_dataset)]

class Combined_M0_Dataset_VCID_Mayo_Train(Combined_M0_Dataset):

    def __init__(self, **kwargs):
        old_dataset = LOFT_M0_Dataset_Train(**kwargs)
        new_dataset = LOFT_M0_Dataset_Train_Mayo(**kwargs)
        super().__init__(old_dataset, new_dataset)

class Combined_M0_Dataset_VCID_Mayo_Valid(Combined_M0_Dataset):

    def __init__(self, **kwargs):
        old_dataset = LOFT_M0_Dataset_Valid(**kwargs)
        new_dataset = LOFT_M0_Dataset_Valid_Mayo(**kwargs)
        super().__init__(old_dataset, new_dataset)
############################################################################################################################################
# CBF datasets
class LOFT_CBF_Dataset_Train(LOFT_CBF_Dataset):

    def __init__(self, AUG_ROT_RANGE = (-60, 60),  **kwargs):
        h5_path = '/ifs/loni/groups/loft/qinyang/M0_generation_data/data_hdf5_csv/M0_generation_ADNI_030524_alldata.hdf5'
        data_conn = h5py.File(h5_path,"r")
        csv_path = '/ifs/loni/groups/loft/qinyang/M0_generation_data/data_hdf5_csv/M0_generation_ADNI_030524_alldata.csv'
        csv = pd.read_csv(csv_path)
        data_splits = {
            "train": csv[csv["train_val_test"] == "TRAIN"]["subject_ID"].tolist(),
        }
        CBF_data = LOFT_M0Data(data_conn, data_splits)        
        train_preproc_lst = [
            AugmentRandomRotationMultiple(AUG_ROT_RANGE),
            AugmentHorizonalFlipMultiple(),        
        ]        
        id_lst = data_splits["train"]
        id_slice_lst = CBF_data._get_id_slice_lst(id_lst)
        super().__init__(data_conn, id_slice_lst, preproc_lst=train_preproc_lst, **kwargs)
    
class LOFT_CBF_Dataset_Valid(LOFT_CBF_Dataset):

    def __init__(self,  **kwargs):
        h5_path = '/ifs/loni/groups/loft/qinyang/M0_generation_data/data_hdf5_csv/M0_generation_ADNI_030524_alldata.hdf5'
        data_conn = h5py.File(h5_path,"r")
        csv_path = '/ifs/loni/groups/loft/qinyang/M0_generation_data/data_hdf5_csv/M0_generation_ADNI_030524_alldata.csv'
        csv = pd.read_csv(csv_path)
        data_splits = {
            "train": csv[csv["train_val_test"] == "TRAIN"]["subject_ID"].tolist(),
            "valid": csv[csv["train_val_test"] == "VALID"]["subject_ID"].tolist(),
        }
        CBF_data = LOFT_M0Data(data_conn, data_splits) 
        id_lst = data_splits["valid"]
        id_slice_lst = CBF_data._get_id_slice_lst(id_lst)       
        super().__init__(data_conn, id_slice_lst,  **kwargs)

class LOFT4DMRPredDataset(Dataset):
    """ Dataset class for dicom (used primarly for inference)
    :params: dicom_path: directory which contains dicom files
    """
    def __init__(self, input_mtx, preproc_lst = None):
        # save data
        self.preproc_lst = preproc_lst if preproc_lst is not None else []

        # save dcm info
        self.pxl_mtx = input_mtx
        # save pxl matrix info
        # move to numpy matrix and apply preprocessing

        self.pxl_mtx = self._preprocess_data(self.pxl_mtx)
        self.pxl_mtx = self.pxl_mtx.astype('float32')


    def _preprocess_data(self, input_mtx):
        """ preprocesses dicom data
        :params: input_mtx: input numpy matrix
        :returns: applied preprocessing steps
        """
        # apply preprocessing steps
        for curr_preproc in self.preproc_lst:
            input_mtx = curr_preproc(input_mtx)[0]

        return input_mtx


    def __len__(self):
        """ calculates total number of steps
        :returns: length of self.data_lst
        """
        return self.pxl_mtx.shape[0]

    def __getitem__(self, indx):
        """ primary getter method for Dataset
        :params: indx: index to return
        :returns: [0] input image
        """
        # get index of image and then
        input_mtx = self.pxl_mtx[indx]

        # expand dims for channel
        input_mtx = np.expand_dims(input_mtx, 0)
        if input_mtx.ndim==4:
            input_mtx = np.squeeze(input_mtx, axis=0)
        # must return tuple
        return input_mtx,