from experiments.experiment import BaselineExperiment
import segmentation_models as sm


def simple_baseline_test():
    model = sm.Unet(encoder_weights='imagenet', activation='sigmoid')
    BaselineExperiment(identifier="01").run(
        dataset_size=1,
        batch_size=1,
        image_resolution=(512, 512),
        model=model,
        epochs=1
    )


if __name__ == '__main__':
    simple_baseline_test()
