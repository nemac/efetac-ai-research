class MaskNotPresentWarning(Warning):
    """Warning for when a corresponding mask for an image file cannot be found by RasterioDataset. Inspired by
    https://www.youtube.com/watch?v=LrtnLEkOwFE&ab_channel=ArjanCodes"""

    def __init__(self, image_path: str, mask_path: str, message: str):
        self.image_path = image_path
        self.mask_path = mask_path
        self.message = message
        super().__init__(message)
