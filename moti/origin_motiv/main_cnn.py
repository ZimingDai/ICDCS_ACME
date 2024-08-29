import argparse
import json

import setproctitle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, AutoConfig

# 重命名进程名
proc_title = "PhoenixDai_Python"
setproctitle.setproctitle(proc_title)


class SimpleCNN(nn.Module):
    def __init__(self, channels, image_size, patch_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.width = self.height = image_size // patch_size
        self.channels = channels

        self.conv = nn.Conv2d(self.channels, 16, kernel_size=3, stride=1, padding=1)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * (self.width // 2) * (self.height // 2), num_classes)

    def forward(self, x):
        # Remove the class token and reshape to (batch_size, channels, width, height)
        x = x[:, 1:].reshape(-1, self.channels, self.width, self.height)

        x = self.conv(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ViTWithAuxiliaryClassifiersCNNIgnore(ViTForImageClassification):
    def __init__(self, config, num_classes=100):
        super().__init__(config)  # 调用父类(ViTForImageClassification)的构造函数
        self.num_classes = num_classes  # 设置分类任务的类别数量

        # Create a main classifier
        self.classifier = SimpleCNN(config.hidden_size, config.image_size, config.patch_size, num_classes)

        # Create auxiliary classifiers for each Transformer layer
        self.auxiliary_classifiers = nn.ModuleList([
            SimpleCNN(config.hidden_size, config.image_size, config.patch_size, num_classes)
            for _ in range(config.num_hidden_layers)
        ])

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, num_classes=100):
        # 加载预训练模型的配置
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        # 创建一个新实例，但还没有加载预训练权重
        model = cls(config, num_classes=num_classes)

        # 加载预训练权重
        pretrained_dict = torch.load(pretrained_model_name_or_path + '/pytorch_model.bin', map_location='cpu')
        # 调整键名以匹配自定义模型中的层名称
        pretrained_dict = {f'vit.{k}': v for k, v in pretrained_dict.items()}

        # 获取模型当前的状态字典
        model_dict = model.state_dict()

        # # # 测试预训练模型与自定义模型的层名称是否对齐
        # # 找出预训练模型中有而自定义模型中没有的层
        # missing_layers = [k for k in pretrained_dict.keys() if k not in model_dict]

        # print("Layers present in pretrained model but missing in custom model:")
        # for layer in missing_layers:
        #     print(layer)

        # # 找出自定义模型中有而预训练模型中没有的层
        # extra_layers = [k for k in model_dict.keys() if k not in pretrained_dict]

        # print("Layers present in custom model but missing in pretrained model:")
        # for layer in extra_layers:
        #     print(layer)

        # 过滤出预训练字典中与模型字典匹配的部分
        adjust_pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                  k in model_dict and model_dict[k].size() == v.size()}

        # 更新现有的模型状态字典
        model_dict.update(adjust_pretrained_dict)

        # 加载更新后的状态字典
        # 这将仅加载存在于预训练模型中的权重，而新添加的部分将保留其初始化状态
        model.load_state_dict(model_dict)

        # ## 测试预训练权重是否加载进去。
        # model_dict = model.state_dict()
        # for key in pretrained_dict:
        #     if key in model_dict:
        #         # 如果两个状态字典中都有这个键，比较它们的权重
        #         are_equal = torch.equal(pretrained_dict[key], model_dict[key])
        #         print(f"Layer {key}: {'Equal' if are_equal else 'Not equal'}")

        return model

    def forward(self, pixel_values, labels=None):
        # 调用ViT模型的前向传播方法，输出隐藏状态和其他信息
        outputs = self.vit(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # 获取所有隐藏层的状态

        # 使用ViT模型的最后一层隐藏状态计算主分类器的输出
        logits = self.classifier(hidden_states[-1])

        aux_logits = [classifier(hidden_state) for classifier, hidden_state in
                      zip(self.auxiliary_classifiers, hidden_states)]

        return logits, aux_logits  # 返回主分类器和所有辅助分类器的输出


class CNNProject(nn.Module):
    def __init__(self, channels, image_size, patch_size, num_classes):
        super(CNNProject, self).__init__()
        self.width = self.height = image_size // patch_size
        self.channels = channels

        self.conv = nn.Conv2d(self.channels * 2, 16, kernel_size=3, stride=1, padding=1)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * (self.width // 2) * (self.height // 2), num_classes)

    def forward(self, x, cls_token):
        # Concatenate the CLS token with every patch
        cls_token = cls_token.unsqueeze(-1).unsqueeze(-1)  # Reshape CLS token to match the spatial dimensions
        x = x[:, 1:].reshape(-1, self.channels, self.width, self.height)
        x = torch.cat([cls_token.expand(-1, -1, self.width, self.height), x], dim=1)
        x = self.conv(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ViTWithAuxiliaryClassifiersCNNProject(ViTForImageClassification):
    def __init__(self, config, num_classes=100):
        super().__init__(config)  # 调用父类(ViTForImageClassification)的构造函数
        self.num_classes = num_classes  # 设置分类任务的类别数量

        # Create a main classifier
        self.classifier = CNNProject(config.hidden_size, config.image_size, config.patch_size, num_classes)

        # Create auxiliary classifiers for each Transformer layer
        self.auxiliary_classifiers = nn.ModuleList([
            CNNProject(config.hidden_size, config.image_size, config.patch_size, num_classes)
            for _ in range(config.num_hidden_layers)
        ])

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, num_classes=100):
        # 加载预训练模型的配置
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        # 创建一个新实例，但还没有加载预训练权重
        model = cls(config, num_classes=num_classes)
        pretrained_dict = torch.load(pretrained_model_name_or_path + '/pytorch_model.bin', map_location='cpu')
        # 调整键名以匹配自定义模型中的层名称
        pretrained_dict = {f'vit.{k}': v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        adjust_pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                  k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(adjust_pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    def forward(self, pixel_values, labels=None):
        # 调用ViT模型的前向传播方法，输出隐藏状态和其他信息
        outputs = self.vit(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # 获取所有隐藏层的状态

        cls_token = hidden_states[-1][:, 0, :]
        # Process each Transformer layer's output
        logits = self.classifier(hidden_states[-1], cls_token)

        aux_logits = [classifier(hidden_state, hidden_state[:, 0, :]) for classifier, hidden_state in
                      zip(self.auxiliary_classifiers, hidden_states)]

        return logits, aux_logits  # 返回主分类器和所有辅助分类器的输出


def main(file_dir):
    # 命令行参数
    parser = argparse.ArgumentParser()

    parser.add_argument("--head_name", default=None, type=str, required=True, help="The type of the header.")

    parser.add_argument("--device", default=1, type=str, help="Choose the device to train.")

    parser.add_argument("--epoch", default=10, type=int, help="Epoch")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")

    parser.add_argument("--seed", default=42, type=int, help="Random Seed")

    args = parser.parse_args()

    # 随机种子设置
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)  # 如果使用多个GPU

    # 打开Tensorboard
    writer = SummaryWriter(f"./runs/{args.head_name}_{args.seed}")

    # 输出本轮次运行信息
    with open('./vit-base-patch16-224/config.json') as f:
        temp_config = json.load(f)

    print(
        f"\nPretrain, {args.head_name}, {temp_config['image_size']}*{temp_config['image_size']}, {temp_config['patch_size']} patch, {args.epoch} epoch",
        file=file_dir)

    # 检查并打印GPU信息
    if torch.cuda.is_available():
        current_device = torch.device(f"cuda:{args.device}")
        print(f"Current GPU: {torch.cuda.get_device_name(current_device)}")
    else:
        print("No GPU available, using CPU instead.")
        current_device = 'cpu'
    # 定义使用的设备
    device = torch.device(current_device)

    # 数据预处理步骤
    transform = transforms.Compose([
        transforms.Resize((temp_config['image_size'], temp_config['image_size'])),  # 调整图像大小为224x224以符合ViT模型输入要求
        transforms.ToTensor(),  # 将图像数据转换为PyTorch张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将图像数据标准化
    ])

    # 加载CIFAR-100训练集
    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # 创建数据加载器用于批量处理和打乱数据

    # 加载CIFAR-100测试集
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)  # 创建测试数据加载器

    # config = AutoConfig.from_pretrained('./vit_base_config')
    # save_directory = "./vit_base_config"
    # # 保存模型配置到本地目录
    # config.save_pretrained(save_directory)
    task_name = args.head_name
    if task_name == "cnn_ignore":
        model = ViTWithAuxiliaryClassifiersCNNIgnore.from_pretrained('./vit-base-patch16-224', num_classes=100)
    elif task_name == "cnn_project":
        model = ViTWithAuxiliaryClassifiersCNNProject.from_pretrained('./vit-base-patch16-224', num_classes=100)
    elif task_name == "cnn_add":
        model = ViTWithAuxiliaryClassifiersCNNIgnore.from_pretrained('./vit-base-patch16-224', num_classes=100)

    model.to(device)

    # 设置优化器，这里使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    # 设置损失函数，这里使用交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 开始训练模型
    model.train()  # 将模型设置为训练模式
    for epoch in range(args.epoch):  # 迭代10个训练周期

        total_correct = 0
        total = 0
        aux_accuracies = [0] * len(model.auxiliary_classifiers)

        for i, batch in enumerate(train_loader):  # 从数据加载器中迭代取出数据
            images, labels = batch  # 获取图像和标签
            images, labels = images.to(device), labels.to(device)  # 将数据移至设备

            # 前向传播：计算模型输出
            logits, aux_logits = model(images)
            loss = criterion(logits, labels)  # 计算主分类器的损失

            # 计算主分类器的准确性
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 计算每个辅助分类器的损失并累加
            for j, aux_logit in enumerate(aux_logits):
                loss += criterion(aux_logit, labels)

                # 计算辅助分类器的准确性
                _, predicted_aux = torch.max(aux_logit, 1)
                aux_accuracies[j] += (predicted_aux == labels).sum().item()

            # 反向传播和优化
            optimizer.zero_grad()  # 清空过去的梯度
            loss.backward()  # 计算损失的梯度
            optimizer.step()  # 根据梯度更新模型参数

            print(f"Epoch {epoch}, Loss: {loss.item()}")  # 打印训练损失
            # 将损失记录到TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

        # 在epoch结束时记录平均准确性
        main_accuracy = total_correct / total
        writer.add_scalar('Accuracy/main_classifier', main_accuracy, epoch)

        for j, acc in enumerate(aux_accuracies):
            aux_accuracy = acc / total
            writer.add_scalar(f'Accuracy/aux_classifier_{j}', aux_accuracy, epoch)

    # 评估模型性能
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 测试阶段不计算梯度
        total = 0  # 记录总样本数
        correct = 0  # 记录主分类器正确预测的样本数
        aux_correct = [0] * len(model.auxiliary_classifiers)  # 记录每个辅助分类器正确预测的样本数
        for batch in test_loader:  # 从数据加载器中迭代取出数据
            images, labels = batch  # 获取图像和标签
            images, labels = images.to(device), labels.to(device)  # 将数据移至设备

            logits, aux_logits = model(images)  # 计算主分类器和辅助分类器的输出
            _, predicted = torch.max(logits, 1)  # 获取主分类器的预测结果
            total += labels.size(0)  # 更新总样本数
            correct += (predicted == labels).sum().item()  # 更新主分类器正确预测的样本数

            for i, aux_logit in enumerate(aux_logits):  # 遍历每个辅助分类器
                _, predicted_aux = torch.max(aux_logit, 1)  # 获取辅助分类器的预测结果
                aux_correct[i] += (predicted_aux == labels).sum().item()  # 更新辅助分类器正确预测的样本数

        print(f"Accuracy on CIFAR-100 test set (Main Classifier): {100 * correct / total}%",
              file=file_dir)  # 打印主分类器的准确率
        for i, correct_aux in enumerate(aux_correct):  # 遍历每个辅助分类器
            # 打印每个辅助分类器的准确率
            print(f"Accuracy on CIFAR-100 test set (Auxiliary Classifier {i}): {100 * correct_aux / total}%",
                  file=file_dir)
    writer.close()


if __name__ == "__main__":
    with open('log/output.txt', 'a') as file:
        main(file)
