import cv2
import numpy as np
import seam_carving as sm


class ImagePreprocessor:
    IMAGE_SIZE = (512, 512)

    def apply_image_default(self, image_path, normalize=True, resize_to=IMAGE_SIZE, seam_carving=False):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if resize_to:
            if seam_carving:
                image = sm.resize(
                    src=image,
                    size=resize_to,
                    energy_mode='forward',
                    order='width-first',
                    keep_mask=None
                )
            else:
                image = cv2.resize(src=image, dsize=resize_to, interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # normalization step to get numbers between (0, 1)
        if normalize:
            image = image / 255.0

        return image

    def apply_mask_default(self, mask_path, normalize=True, resize_to=IMAGE_SIZE, seam_carving=False):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        no_colors = len(np.unique(mask))
        if resize_to:
            if seam_carving:
                mask = sm.resize(
                    src=mask,
                    size=resize_to,
                    energy_mode='backward',
                    order='width-first',
                    keep_mask=None
                )
            else:
                mask = cv2.resize(src=mask, dsize=resize_to, interpolation=cv2.INTER_NEAREST)

        # normalization step to get numbers between (0, 1)
        if normalize:
            mask = mask / 255.0

        # binarize mask if more that 2 unique colors
        if no_colors > 2:
            mask = np.around(mask)
        mask = np.expand_dims(mask, axis=2)

        # assert mask has only two distinct colors
        assert len(np.unique(mask)) == 2

        return mask
