from torchvision import datasets

if __name__ == '__main__':
    dataset = datasets.MNIST('.', train=True, download=True)
    dataset = datasets.MNIST('.', train=False, download=True)
