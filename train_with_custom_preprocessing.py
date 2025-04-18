import sys
import os
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# 🔼 Add this to fix the import issue
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'run'))

from run import DeepSVDD
from dataset.main import load_dataset

def custom_transforms():
    transform = transforms.Compose([
        transforms.Resize((32, 32), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
        transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x))
    ])
    return transform

def main():
    dataset_name = 'cifar10'
    normal_class = 3
    data_path = './data'
    export_path = './log/custom_run'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading dataset...")
    dataset = load_dataset(dataset_name, data_path, normal_class, custom_transforms())

    print("Initializing Deep SVDD...")
    deep_svdd = DeepSVDD(objective='one-class', nu=0.1)
    deep_svdd.set_network('cifar10_LeNet')

    # Optional pretraining
    deep_svdd.pretrain(dataset,
                       optimizer_name='adam',
                       lr=0.0001,
                       n_epochs=10,
                       batch_size=128,
                       weight_decay=1e-6,
                       device=device)

    print("Training Deep SVDD...")
    deep_svdd.train(dataset,
                    optimizer_name='adam',
                    lr=0.0001,
                    n_epochs=50,
                    lr_milestone=30,
                    batch_size=128,
                    weight_decay=1e-6,
                    device=device,
                    print_epoch=5)

    print("Testing...")
    deep_svdd.test(dataset, device=device)

    print(f"Training complete. Results saved to: {export_path}")

if __name__ == '__main__':
    main()
