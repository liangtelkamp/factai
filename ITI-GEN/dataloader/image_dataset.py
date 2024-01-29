import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class ImgDataset(Dataset):
    """
    Construct the dataset
    """
    def __init__(self, root_dir, label, upper_bound=200):
        self.root_dir = root_dir
        self.label = label
        self.upper_bound = upper_bound

        self.file_list = []
        self.prepare_data_list()

        self.transforms = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def prepare_data_list(self):
        # set all images together
        # all into self.file_list
        for i, dir in enumerate(self.root_dir):
            path = glob.glob(os.path.join(dir, '*.jpg'))
            path += glob.glob(os.path.join(dir, '*.png'))

            # random select the images based on the min(upper, size)
            for index in np.random.choice(len(path), min(self.upper_bound,len(path)), replace=False):
                self.file_list.append({'path': path[index], 'label': self.label[i]})
        self.file_list = [{'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/075239.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/013060.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/018548.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/024290.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/025154.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/027094.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/031365.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/033895.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/042083.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/042891.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/046695.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/048449.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/057701.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/059904.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/019265.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/020024.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/024349.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/027101.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/027165.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/027806.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/028589.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/029465.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/029623.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/029685.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/positive/030484.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/180238.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/186491.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/100616.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/106863.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/063491.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/127948.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/055948.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/106358.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/134914.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/119236.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/084934.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/050101.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/041583.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/071816.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/106538.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/026107.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/041072.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/101834.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/065840.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/124629.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/046271.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/000703.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/166781.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/038848.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Eyeglasses/negative/161909.jpg', 'label': 1}]

        print(f"Self file list : {(self.file_list)}")


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx]['path'])
        img = self.transforms(img)
        label = self.file_list[idx]['label']
        return dict(img=img, label=label)