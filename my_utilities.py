import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
# from tqdm.notebook import tqdm
# from tqdm import tqdm_notebook as tqdm

class StatsOneLayer:
    def __init__(self, model, n_classes, data_loader, bins_size=2,
                 entropy_calculation=True, mi_calculation=True, kl_calculation=True):
        self.bin_size = bins_size
        self.n_classes = n_classes
        self.__calculate_distribution(model, n_classes, data_loader)
        if entropy_calculation:
            self.__calculate_entropy()
        if mi_calculation:
            self.__calculate_mi()
        if kl_calculation:
            self.__calculate_kl()

    def __calculate_distribution(self, model, n_classes, data_loader):
        x, _ = next(iter(data_loader))
        output_shape = model(x).shape[1:]
        self.joint_distribution = np.zeros((*output_shape, 2, 10))
        self.layer_mean = np.zeros(output_shape)
        self.layer_sqr_mean = np.zeros(output_shape)
        for x, target in data_loader:  # data_loader
            output = model(x).detach().cpu().numpy()
            self.layer_mean += 1 / len(data_loader) * output.mean(0)
            self.layer_sqr_mean += 1 / len(data_loader) * (output ** 2).mean(0)
            # P(bin | target) calculation
            targets, count = np.unique(target, return_counts=True)
            targets_count = np.zeros(10)
            targets_count[targets] = count
            self.joint_distribution[..., 0, :] += np.multiply((output <= 0)[..., None].sum(0), targets_count[None, :])
            self.joint_distribution[..., 1, :] += np.multiply((output > 0)[..., None].sum(0), targets_count[None, :])
        self.joint_distribution /= self.joint_distribution.sum(-1, keepdims=True).sum(-2, keepdims=True)

    def __calculate_entropy(self):
        bins_distribution = self.joint_distribution.sum(-1)
        bins_distribution /= bins_distribution.sum(-1, keepdims=True)
        self.entropy = -np.sum(bins_distribution * np.nan_to_num(np.log2(bins_distribution), neginf=0), axis=-1)

    def __calculate_mi(self):
        target_distribution = self.joint_distribution.sum(-2)
        target_distribution /= target_distribution.sum(-1, keepdims=True)
        bins_distribution = self.joint_distribution.sum(-1)
        bins_distribution /= bins_distribution.sum(-1, keepdims=True)
        pairwise_distribution = np.einsum('...j,...k->...jk', bins_distribution, target_distribution)
        self.mi = (self.joint_distribution *
                   np.nan_to_num(np.log2(self.joint_distribution / pairwise_distribution), neginf=0)).sum(-1).sum(-1)

    def __calculate_kl(self):
        target_conditional_distribution = self.joint_distribution / self.joint_distribution.sum(-2, keepdims=True)
        bins_distribution = self.joint_distribution.sum(-1, keepdims=True)
        bins_distribution /= bins_distribution.sum(-2, keepdims=True)
        self.kl = (target_conditional_distribution * np.nan_to_num( \
            np.log2(target_conditional_distribution) - np.log2(bins_distribution)), neginf=0)).sum(-2)

def load_data(dataset_name='CIFAR'):
    # configured for datasets available in torchvision.datasets for example MNIST and FashionMNIST. For other datasets do appropriate modifications
    dataset_name = dataset_name
    dataset_method = getattr(torchvision.datasets, dataset_name)
    PATH = "./data/" + dataset_name + '/'
    normalization_mean = (0.5, 0.5, 0.5)
    normalization_std = (0.5, 0.5, 0.5)
    # batch sizes
    valid_split = 0.2
    batch_size_train = 32
    batch_size_val = 1024
    batch_size_test = 1024

    # Loading the dataset.
    dataset = dataset_method(f'{PATH}trn/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(normalization_mean, normalization_std)]))

    # This segment of code splits the training dataset into training and validation datasets. We use the .Subset class to split the dataset using randomly shuffled indices.
    dataset_size = len(dataset)
    split = int(np.floor(valid_split * dataset_size))
    # Using a fixed seed so that same split happens for all parallel instances of independent simulations
    np.random.seed(42)
    indices = np.random.permutation(np.arange(dataset_size))

    train_idx, valid_idx = indices[split:], indices[:split]

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    valid_dataset = torch.utils.data.Subset(dataset, valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size_train, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size_val, shuffle=False)

    test_dataset = dataset_method(f'{PATH}test/', train=False, download=True,
                                  transform=torchvision.transforms.Compose([
                                      torchvision.transforms.ToTensor(),
                                      torchvision.transforms.Normalize(
                                          normalization_mean, normalization_std)]))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size_test, shuffle=False)
    return train_loader, valid_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion, device='cpu'):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def calc_acc(model, data_loader, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        acc = 0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            acc += (predicted == labels).sum().item()
    return acc / len(data_loader.dataset)