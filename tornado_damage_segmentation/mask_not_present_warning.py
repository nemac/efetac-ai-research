class MaskNotPresentWarning(UserWarning):
    """Warning for when a corresponding mask for an image file cannot be found by RasterioDataset. Inspired by
    https://www.youtube.com/watch?v=LrtnLEkOwFE&ab_channel=ArjanCodes"""

    def __init__(self, message: str):
        self.message = message
