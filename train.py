import os
import sys
import json
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import resnet34
from torch.utils.data import DataLoader

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 数据增强
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 获取当前目录的绝对路径
    data_root = os.getcwd()
    # 获取数据集路径
    image_path = os.path.join(data_root, 'data_set', 'flower_data')
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # 数据集加载
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                         transform=data_transform['train'])
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                                            transform=data_transform['val'])
    val_num = len(validate_dataset)
    train_num = len(train_dataset)
    # 按文件夹的名字来确定的类别
    # train_dataset返回一个字典, {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    # 调换键值
    class_dict = dict((v, k) for k, v in flower_list.items())
    # 将该字典保存为json文件
    # 将json文件格式化，indent设置每个k:v的左缩进格
    json_str = json.dumps(class_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True,
                              num_workers=nw)
    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=batch_size, shuffle=True,
                                 num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    net = resnet34()
    # 加载预训练模型权重
    model_weight_path = 'resnet34_pre.pth'
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    net.fc = nn.Linear(net.fc.in_features, 5)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    epochs = 3
    best_acc = 0.
    save_path = 'resnet34.pth'

    for epoch in range(epochs):
        # train
        net.train()
        train_bar = tqdm(train_loader)
        for img, label in train_bar:
            logits = net(img.to(device))
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # validate
        net.eval()
        acc = 0.
        test_loss = 0.
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for img, label in val_bar:
                logits = net(img.to(device))
                loss = criterion(logits, label)
                test_loss += loss.item()
                pred = torch.max(logits, dim=1)[1]
                acc += torch.eq(pred, label.to(device)).sum().item()
                val_bar.desc = "validate epoch[{}/{}]".format(epoch + 1,
                                                             epochs)
        val_acc = acc / val_num
        print('[epoch {}] test_loss: {:.3f}  val_accuracy: {:.3f}'.format(
              epoch + 1, test_loss / len(validate_loader), val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
    print("finished training!")

if __name__ == '__main__':
    main()