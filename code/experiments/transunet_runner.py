import numpy as np
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from experiments.transunet_experiment import TransUNetExperiment
from transformers.trans_u_net.backbones.vit_seg_modeling import CONFIGS, VisionTransformer

"""
Image size ablation study with simple baseline setting with transfer learning (trains faster).

Resolutions:
- 224x224
- 192x256
"""


def get_model(vit_name, patch_size=16, image_size=(224, 224)):
    config_vit = CONFIGS[vit_name]
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(image_size[0] / patch_size), int(image_size[1] / patch_size)
        )
    model = VisionTransformer(config_vit, img_size=image_size[0], num_classes=config_vit.n_classes)
    # load backbone (pretrained on ImageNet)
    model.load_from(weights=np.load(config_vit.pretrained_path))
    return model


def get_callbacks():
    return [
        EarlyStopping(
            monitor='val_loss',
            min_delta=1e-5,
            verbose=True,
            patience=8
        ),
        ModelCheckpoint(
            monitor='val_loss',
            verbose=True,
            save_weights_only=False,
            mode='min'
        )
    ]


def experiment_test():
    model = get_model(vit_name="R50-ViT-B_16")
    TransUNetExperiment(
        identifier="transunet_00",
        model=model,
        epochs=2,
        dataset_size=1,
        callbacks=get_callbacks()
    ).run()


def experiment_id_50():
    model = get_model(vit_name="R50-ViT-B_16")
    TransUNetExperiment(
        identifier="transunet_50",
        model=model,
        callbacks=get_callbacks()
    ).run()


if __name__ == '__main__':
    experiment_test()
    # experiment_id_50()
