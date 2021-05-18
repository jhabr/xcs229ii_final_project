from experiments.experiment import BaselineExperiment
import segmentation_models as sm

"""
Image size ablation study with simple baseline setting with transfer learning (trains faster).

Resolutions:
- 128x128
- 192x256
"""


def experiment_test():
    model = sm.Unet(activation='sigmoid')
    BaselineExperiment(identifier="unet_00").run(
        dataset_size=1,
        batch_size=1,
        image_resolution=(192, 256),
        model=model,
        epochs=1
    )


def experiment_id_10():
    model = sm.Unet(encoder_weights='imagenet', activation='sigmoid')
    BaselineExperiment(identifier="unet_10").run(
        batch_size=16,
        image_resolution=(128, 128),
        model=model
    )


def experiment_id_11():
    model = sm.Unet(encoder_weights='imagenet', activation='sigmoid')
    BaselineExperiment(identifier="unet_11").run(
        batch_size=16,
        image_resolution=(256, 192),
        model=model
    )


if __name__ == '__main__':
    experiment_test()
    # experiment_id_10()
    # experiment_id_11()
