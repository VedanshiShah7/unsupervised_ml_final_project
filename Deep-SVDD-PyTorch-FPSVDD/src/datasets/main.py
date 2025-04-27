from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset


def load_dataset(dataset_name, data_path, normal_class, use_custom_transforms=False):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class,use_custom_transforms=use_custom_transforms)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class,use_custom_transforms=use_custom_transforms)

    return dataset
