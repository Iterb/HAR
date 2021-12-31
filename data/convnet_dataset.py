import tensorflow as tf
from tensorflow import keras
import numpy as np


from concurrent.futures import ThreadPoolExecutor


class ConvnetDataset:
    """Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """

    def __init__(
        self,
        x_data,
        y_data,
        linspace_size,
    ):
        self.x_data = x_data[x_data.columns[1]].to_numpy()
        self.y_data = y_data[y_data.columns[1]].to_numpy()
        self.linspace_size = linspace_size

    def __getitem__(self, i):

        # read data

        image = np.array(self.x_data[i])
        indices = np.linspace(0, image.shape[0] - 1, self.linspace_size, dtype=np.int)
        image = image[indices]
        image = np.transpose(image, (2, 1, 0))
        target = np.array(self.y_data[i]).astype("float32")
        target = tf.keras.utils.to_categorical(target, 11)
        return image.astype("float32"), target

    def __len__(self):
        return len(self.y_data)


def default_collate_fn(samples):
    X = np.array([sample[0] for sample in samples])
    Y = np.array([sample[1] for sample in samples])

    return X, Y


class ConvnetDataloader(keras.utils.Sequence):
    """Load data from dataset and form batches
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(
        self,
        dataset,
        collate_fn=default_collate_fn,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        replacement: bool = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.replacement = replacement
        self.indices = []
        self.collate_fn = collate_fn
        self.on_epoch_end()

    def __getitem__(self, index):

        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        samples = []
        if self.num_workers == 0:
            for i in indices:
                data = self.dataset[i]
                samples.append(data)
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for sample in executor.map(lambda i: self.dataset[i], indices):
                    samples.append(sample)
        X, Y = self.collate_fn(samples)
        return X, Y

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.dataset) / self.batch_size))

    def on_epoch_end(self):
        n = len(self.dataset)
        seq = np.arange(0, n)
        if self.shuffle:
            if self.replacement:
                self.indices = np.random.randint(
                    low=0, high=n, size=(n,), dtype=np.int64
                ).tolist()
            else:
                np.random.shuffle(seq)
                self.indices = seq
        else:
            self.indices = seq
