'''
Task: Take in a stack of MRI images and convert them to point cloud based on thresholding voxel intensity

Information about Medical Segmentation Decathalon (MSD) Brain Tumor dataset:
- Download dataset here: http://medicaldecathlon.com/
- Voxels size = (1,1,1) in mm
- Time Unit = sec
- Modality: (CuNeRF paper used T1w (1)) (4th Dimension of array)
    0=FLAIR     1=T1w
    2=t1gd      3=T2w
- labels:0=background, 1=edema, 2=non-enhancing tumor, 3=enhancing tumour
- Size of Training Image = (240, 240, 155, 4)
- Size of Training Label = (240, 240, 155)
- Size of Testing Images = (240, 240, 155, 4)

Information about Kits19 Kidney Tumor datatset:
- Download dataset here: https://github.com/neheller/kits19
- 300 cases (first 209 cases have segmenation labels, last 91 cases do not have segmentation labels)

'''
from torch.utils.data import Dataset
import os
import numpy as np
import json
import nibabel as nib
import math
from pathlib import Path

class MRIDataset(Dataset):
    def __init__(self, dataset_dir, split = 'training', modality = 1):
        self.dataset_dir = dataset_dir
        assert split in ('training', 'test')
        self.split = split

        self.filepths = json.load(open(os.path.join(dataset_dir,'dataset.json')))[split]

        self.modality = modality
        
    def __len__(self):
        return len(self.filepths)
        
    def __getitem__(self, idx):
        if self.split == 'training':
            filepth = self.dataset_dir + self.filepths[idx]['image'][1:]
        elif self.split == 'test':
            filepth = self.dataset_dir + self.filepths[idx][1:]
        vol = nib.load(filepth)
        vol = np.array(vol.dataobj[:,:,:,self.modality])
        vol = vol / np.max(vol) #normalize voxel intensities between 0 and 1
        return vol
        
class CTDataset(Dataset):
    #Taken from kits19 
    def get_full_case_id(self, cid):
        try:
            cid = int(cid)
            case_id = "case_{:05d}".format(cid)
        except ValueError:
            case_id = cid

        return case_id

    def get_case_path(self, cid):
        # Resolve location where data should be living
        data_path = Path(__file__).parent / self.dataset_dir / "data" 
        if not data_path.exists():
            raise IOError(
                "Data path, {}, could not be resolved".format(str(data_path))
            )

        # Get case_id from provided cid
        case_id = self.get_full_case_id(cid)

        # Make sure that case_id exists under the data_path
        case_path = data_path / case_id
        if not case_path.exists():
            raise ValueError(
                "Case could not be found \"{}\"".format(case_path.name)
            )

        return case_path
    
    def load_volume(self, cid):
        case_path = self.get_case_path(cid)
        vol = nib.load(str(case_path / "imaging.nii.gz"))
        return vol


    def load_segmentation(self, cid):
        case_path = self.get_case_path(cid)
        seg = nib.load(str(case_path / "segmentation.nii.gz"))
        return seg


    def load_case(self, cid):
        vol = self.load_volume(cid)
        seg = self.load_segmentation(cid)
        return vol, seg
    
    def __init__(self, dataset_dir, split = 'training'):
        self.dataset_dir = dataset_dir
        assert split in ('training', 'testing')
        self.split = split

    def __len__(self):
        if self.split == 'training':
            return 210
        elif self.split == 'testing':
            return 300-210
        
    def __getitem__(self, idx):
        if self.split == 'training' and 0 <= idx < 210:
            vol = self.load_volume(idx)
            vol = np.array(vol.dataobj[:,:,:])
            vol = vol / np.max(vol) #normalize voxel intensities between 0 and 1
        elif self.split == 'testing' and 210 <= idx < 300:
            vol = self.load_volume(idx)
            vol = np.array(vol.dataobj[:,:,:])
            vol = vol / np.max(vol) #normalize voxel intensities between 0 and 1
        else:
            raise IndexError(
                'index not in split'
            )
        return vol #.transpose(1,2,0)

    
def getPCD(vol, threshold = 0.5):
    return np.transpose(np.where(vol > threshold))

def getIntensity(pcd, vol):
    return vol[pcd[:,0], pcd[:,1], pcd[:,2]]

