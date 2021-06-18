import os
import io
import cv2
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.transforms as transforms


class CustomDataset(Dataset):

    def __init__(self, root_dir, meta_file):
        super(CustomDataset, self).__init__()
        self.root_dir = root_dir
        self.meta_file = meta_file
        with open(meta_file) as f:
            lines = f.readlines()
        self.anno = []
        for line in lines:
            content = json.loads(line)
            self.anno.append(content)
        self.transform = transforms.Compose([
            transforms.Resize(size=448),
            transforms.CenterCrop(size=384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        curr_anno = self.anno[idx]
        filename = os.path.join(self.root_dir, curr_anno['filename'])
        label = int(curr_anno['label'])
        #import pdb
        #pdb.set_trace()
        
        #img_bytes = np.fromfile(filename, dtype=np.uint8)
        #buff = io.BytesIO(img_bytes)
        #with Image.open(buff) as img:
        #    img = img.convert('RGB')
        img = cv2.imread(filename)
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        #img = Image.open(filename)
        img = self.transform(img)
        return {'image': img, 'label': label}


class CustomDataLoader(DataLoader):

    def __init__(self, dataset, batch_size, num_workers):
        super(CustomDataLoader, self).__init__(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    