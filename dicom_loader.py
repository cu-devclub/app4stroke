import numpy as np
import cv2
import pydicom
import torch
from smart_open import smart_open
from torch.utils.data import Dataset, DataLoader
from albumentations import  Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from utils.dicom_utils import is_dicom, get_plane, extract_multiwindow_image

torch.set_num_threads(1)
cv2.setNumThreads(1)

NUM_WORKERS = 2
SIZE = 256
INTER = cv2.INTER_CUBIC
MEAN = [0.4984]
SD = [0.2483]
WINDOWS = ((60, 40), (40, 40), (40, 80))  # (blood, tissue, brain) *start with blood for model compatibility


def gen_ct_array(dicom_paths, windows=WINDOWS):
    images = []
    for file_path in dicom_paths:
        if is_dicom(file_path):
            try:
                data = pydicom.dcmread(smart_open(file_path))
                if get_plane(data) == 'Axial':
                    images.append(extract_multiwindow_image(data, windows))
            except:
                pass

    if len(images) > 0:
        ct_array = np.stack([img for img, _ in sorted(images, key=lambda x: x[1])], axis=0)
    else:
        ct_array = np.array([])
        
    return ct_array


def make_transform(size=SIZE, inter=INTER):
    trans = Compose([
                     Resize(size, size, interpolation=inter),
                     Normalize(MEAN, SD),
                     ToTensorV2(),
                    ])
    return trans


def get_flip_ct_array_loader(dicom_paths, windows=WINDOWS, batch_size=16, shuffle=False):
    X = gen_ct_array(dicom_paths, windows=windows)
    transform = make_transform(size=SIZE, inter=INTER)
    ds = FlipBrainCTArray(X, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS)


class FlipBrainCTArray(Dataset):
    def __init__(self, X, transform=None):
        super(FlipBrainCTArray, self).__init__()
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        img_f = img[:, ::-1]
        img = np.concatenate([img, img_f], axis=2)

        if self.transform:
            _res = self.transform(image=img)
            img = _res['image']

        return img

# # FOR TEST
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     dicom_dir = r'D:\ai4strokesubtypes\Benchmark\1151438'
#     data_loader = get_flip_ct_array_loader(dicom_dir)
#     print(data_loader.dataset.X.shape)
#     for x in data_loader:
#         print(x.shape)
#     fig, axs = plt.subplots(x.shape[1], 1)
#     for i in range(x.shape[1]):
#         axs[i].imshow(x[10][i])
#     plt.show()

