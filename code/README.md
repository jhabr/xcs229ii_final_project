# XCS229ii: Final Project

## Azure

### Helpful commands

SSH Terminal:
```bash
$ ssh -p 61999 scpdxcs@ml-lab-2c26d21d-2e78-44d8-a0be-f41571a9502f.southcentralus.cloudapp.azure.com
```

SCP files to instance:
```bash
$ scp -P 61999 data.zip scpdxcs@ml-lab-2c26d21d-2e78-44d8-a0be-f41571a9502f.southcentralus.cloudapp.azure.com:/home/scpdxcs/projects/xcs229ii_final_project/code/data
```

### Baseline
#### Environment Setup

Activate conda environment:
```bash
$ conda activate py37_tensorflow
```

Install pip packages:
```bash
$ pip3 install segmentation_models
$ pip3 install albumentations
```

Export variables:
```bash
$ export SM_FRAMEWORK=tf.keras
$ export PYTHONPATH=/home/scpdxcs/projects/xcs229ii_final_project/code
```

#### Training

Run baseline training:
```bash
$ bash scripts/run_baseline_experiments.sh
```

### TransUNet
#### Environment Setup

Activate conda environment:
```bash
$ conda activate py37_pytorch
```

Install pip packages:
```bash
$ pip3 pip install pytorch-lightning
```

#### Training

Run baseline training:
```bash
$ bash scripts/run_transunet_experiments.sh
```