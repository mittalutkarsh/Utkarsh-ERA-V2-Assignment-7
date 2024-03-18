import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


SEED = 1

def get_dataloaders(data_dir, batch_size, num_workers, pin_memory):
    simple_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                        ])
    test_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                        ])

    exp_train = datasets.MNIST(data_dir, train=True, download=True, transform=simple_transforms)
    exp_test = datasets.MNIST(data_dir, train=False, download=True, transform=test_transforms)

    exp_data_train = exp_train.data

    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader = torch.utils.data.DataLoader(exp_train, **dataloader_args)


    #exp_test = datasets.MNIST(data_dir, train=True, download=True, transform=test_transforms)
    
    #exp_test = datasets.MNIST(data_dir, train=True, download=True, transform=simple_transforms)
    test_loader = torch.utils.data.DataLoader(exp_test, **dataloader_args)

    return train_loader, test_loader, exp_data_train, exp_test,test_loader
