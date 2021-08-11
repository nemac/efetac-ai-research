import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio


class RasterioDataset(Dataset):
    """Defines a dataset for reading rasterio objects. Relies on corresponding images and masks having the same
    filename and the mask directory being flat (no subfolders)"""

    def __init__(self, image_path, mask_path, required_extension='.tif'):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.required_extension = required_extension
        self.rasters = []
        self.get_rasters()

    def __len__(self):
        return len(self.rasters)

    def __getitem__(self, index):
        return [torch.tensor(raster.read(1)) for raster in self.rasters[index]]

    # Generates tuples of data, mask pairs
    def get_rasters(self):
        for _, _, files in os.walk(self.image_path):
            for file in files:
                if file[-3] == self.required_extension:
                    self.rasters.append((rasterio.open(os.path.join(self.image_path, file),
                                                       rasterio.open(os.path.join(self.mask_path, file)))))

    @staticmethod
    # Specifies how the data loader should batch samples
    def collate(batch):
        return torch.tensor(np.array(batch))
