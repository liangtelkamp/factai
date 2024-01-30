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
        self.file_list = [{'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/003113.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/106754.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/006308.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/007591.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/007745.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/008551.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/009774.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/010895.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/015850.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/016638.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/017085.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/017130.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/018643.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/005689.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/006561.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/007201.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/009313.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/015389.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/019631.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/022482.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/026853.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/027112.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/035702.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/037122.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/positive/038883.jpg', 'label': 0}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/000925.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/007102.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/017616.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/023549.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/026327.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/039929.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/040407.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/042306.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/043028.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/044550.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/048306.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/048864.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/050773.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/000223.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/000452.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/000926.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/003109.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/006251.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/006388.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/007819.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/008007.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/008418.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/009901.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/012407.jpg', 'label': 1}, {'path': '/content/drive/My Drive/FACT_AI/data/celeba/Young/negative/013964.jpg', 'label': 1}]

        print(f"Self file list : {(self.file_list)}")


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx]['path'])
        img = self.transforms(img)
        label = self.file_list[idx]['label']
        return dict(img=img, label=label)