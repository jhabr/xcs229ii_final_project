import matplotlib.pyplot as plt


class Visualisation:

    def plot_images(self, **images):
        """PLot images in one row."""
        n = len(images)
        plt.figure(figsize=(16, 5))
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
        plt.show()

    def plot_curves(self, history):
        plt.figure(figsize=(30, 15))
        plt.subplot(221)
        plt.plot(history.history['iou_score'])
        plt.plot(history.history['val_iou_score'])
        plt.title('IoU Score')
        plt.ylabel('iou_score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(222)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # precision & recall
        plt.subplot(223)
        plt.plot(history.history['precision'])
        plt.plot(history.history['val_precision'])
        plt.title('Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.subplot(224)
        plt.plot(history.history['recall'])
        plt.plot(history.history['val_recall'])
        plt.title('Recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
