from baseline.trainer import Trainer

if __name__ == '__main__':
    history = Trainer().train_from_simple_dataloader(dataset_size=10)
