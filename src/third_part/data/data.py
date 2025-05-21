from torchvision import datasets, transforms
from data.sampling import noniid_share_specified_category, data_test
import torch

def get_dataset_distributed(args):
    if args.dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
        ])

        train_dataset = datasets.CIFAR10('/data/gaofei/project/Taylor_pruning-master/data/cifar10', train=True, download=True,
                                       transform=transform_train)

        test_dataset = datasets.CIFAR10('/data/gaofei/project/Taylor_pruning-master/data/cifar10', train=False, download=True,
                                      transform=transform_test)
        user_groups, label_user_groups = noniid_share_specified_category(args, train_dataset, args.num_users,
                                                                         args.dataset)
        test_user_groups = data_test(args, test_dataset, args.num_users, label_user_groups)

    elif args.dataset == "cifar100":
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        train_dataset = datasets.CIFAR100('/data/gaofei/project/Taylor_pruning-master/data/cifar100', train=True, download=True,
                                          transform=transform_train)

        test_dataset = datasets.CIFAR100('/data/gaofei/project/Taylor_pruning-master/data/cifar100', train=False, download=True,
                                         transform=transform_test)
        user_groups, label_user_groups = noniid_share_specified_category(args, train_dataset, args.num_users,
                                                                         args.dataset)
        test_user_groups = data_test(args, test_dataset, args.num_users, label_user_groups)

    return train_dataset, test_dataset, user_groups, test_user_groups