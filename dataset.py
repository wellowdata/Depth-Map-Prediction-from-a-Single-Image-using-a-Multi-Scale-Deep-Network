import torch
import h5py
import io


from PIL import Image
import numpy as np
from google.cloud import storage
from torch.utils import data


class NYUDataset(data.Dataset):
    def __init__(self, data_dir, tfms):
        super(NYUDataset, self).__init__()
        self.data_dir = data_dir
        self.tfms = tfms
        
        self.ds_v_1 = h5py.File(self.data_dir+'nyu_depth_data_labeled.mat')
        self.ds_v_2 = h5py.File(self.data_dir+'nyu_depth_v2_labeled.mat')
        
        self.len = len(self.ds_v_1["images"]) + len(self.ds_v_2["images"])

           
    def __getitem__(self, index):
        if(index<len(self.ds_v_1["images"])):
            ds = self.ds_v_1    
            i = index
        else:    
            ds = self.ds_v_2
            i = index - len(self.ds_v_1["images"])

        img = np.transpose(ds["images"][i], axes=[2,1,0])
        img = img.astype(np.uint8)

        depth = np.transpose(ds["depths"][i], axes=[1,0])
        depth = (depth/depth.max())*255
        depth = depth.astype(np.uint8)

        if self.tfms:
            tfmd_sample = self.tfms({"image":img, "depth":depth})
            img, depth = tfmd_sample["image"], tfmd_sample["depth"]
        return (img, depth)
    
    def __len__(self):
        return self.len    

    
class ADE20kDataset(data.Dataset):
    def __init__(self, data_dir, tfms):
        super(ADE20kDataset, self).__init__()
        #self.data_dir = data_dir
        self.gen = ADE20k_iterator()
        self.tfms = tfms
        
        #self.ds_v_1 = h5py.File(self.data_dir+'nyu_depth_data_labeled.mat')
        #self.ds_v_2 = h5py.File(self.data_dir+'nyu_depth_v2_labeled.mat')
        
        #self.len = len(self.ds_v_1["images"]) + len(self.ds_v_2["images"])

           
    def __getitem__(self, index):
#         if(index<len(self.ds_v_1["images"])):
#             ds = self.ds_v_1    
#             i = index
#         else:    
#             ds = self.ds_v_2
#             i = index - len(self.ds_v_1["images"])
        img, filepath = next(self.gen)
        img = np.transpose(img, axes=[2,1,0])
        img = img.astype(np.uint8)

        depth = None #np.transpose(ds["depths"][i], axes=[1,0])
        #depth = (depth/depth.max())*255
        #depth = depth.astype(np.uint8)

        if self.tfms:
            tfmd_sample = self.tfms({"image":img, "depth":None})  #this is a major fudge to fudge it until it worked
            img, depth = tfmd_sample["image"], tfmd_sample["depth"]  
        return (img, depth)
    
    def __len__(self):
        count=0
        client = storage.Client()
        bucket = client.get_bucket('ucl-interior-desing')
        prefix='ADE20K'
        #/ADE20K_2016_07_26/images/validation/b/ballroom/'
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            if blob.name.endswith('.jpg'):
                count +=1
        return count    

    
def ADE20k_iterator():
    client = storage.Client()
    bucket = client.get_bucket('ucl-interior-desing')
    prefix='ADE20K'
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        if blob.name.endswith('.jpg'):
            contenido = blob.download_as_string()
            fp = io.BytesIO(contenido)
            img = Image.open(fp)
            yield img, blob.name
            
            
if __name__ == "__main__":
    from custom_transforms import *
    bs = 8
    sz = (320,240)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    mean, std = torch.tensor(mean), torch.tensor(std)

    test_tfms = transforms.Compose([
        transforms.Resize((240,320)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    #ds = NYUDataset('data/', tfms)
    ds = ADE20kDataset('', test_tfms)
    dl = torch.utils.data.DataLoader(ds, bs, shuffle=False)
    img, depth = iter(dl).next()
    print(img)