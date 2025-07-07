import torch
from torchvision import datasets, transforms

def save_fashionmnist_txt(train_file="fashion_train.txt", test_file="fashion_test.txt", n_train=60000, n_test=10000):
    transform = transforms.Compose([
        transforms.ToTensor()  # Output: [1, 28, 28], values in [0, 1]
    ])

    train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    def write_file(dataset, path, n_samples):
        with open(path, "w") as f:
            n_inputs = 28 * 28  # = 784
            n_outputs = 10
            f.write(f"{n_inputs} {n_outputs}\n")
            for i in range(n_samples):
                image, label = dataset[i]  # image: [1, 28, 28]
                image_flat = image.view(-1).numpy()  # [784]
                image_str = ' '.join(f"{x:.6f}" for x in image_flat)
                label_str = ' '.join(['1' if j == label else '0' for j in range(10)])
                f.write(f"{image_str} {label_str}\n")

    write_file(train_dataset, train_file, n_train)
    write_file(test_dataset, test_file, n_test)

if __name__ == "__main__":
    save_fashionmnist_txt("fashion_train.txt", "fashion_test.txt", 60000, 10000)
