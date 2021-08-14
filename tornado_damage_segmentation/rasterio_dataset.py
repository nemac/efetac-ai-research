import os
import torch
from torch.utils.data import Dataset
import rasterio


class RasterioDataset(Dataset):
    """Defines a dataset for reading rasterio objects. Relies on corresponding images and masks having the same
    filename and the mask directory being flat (no subfolders)."""

    def __init__(self, image_path, mask_path):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.rasters = []
        self.load_rasters()

    def __len__(self):
        return len(self.rasters)

    def __getitem__(self, index):
        return tuple([torch.tensor(raster.read(1)) for raster in self.rasters[index]])

    # Generates tuples of data, mask pairs
    def load_rasters(self):
        for _, _, files in os.walk(self.image_path):
            for file in files:
                self.rasters.append((rasterio.open(os.path.join(self.image_path, file)),
                                     rasterio.open(os.path.join(self.mask_path, file))))
