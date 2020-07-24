from torchvision import datasets

if __name__ == '__main__':
    dataset = datasets.MNIST('mnist', train=True, download=True)
    dataset = datasets.MNIST('mnist', train=False, download=True)
