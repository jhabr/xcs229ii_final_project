import albumentations as A


class DataAugmentation:

    def get_preprocessing(self, preprocessing_fn):
        """Construct preprocessing transform

        Args:
            preprocessing_fn (callbale): data normalization function
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose

        """
        _transform = [A.Lambda(image=preprocessing_fn)]
        return A.Compose(_transform)
