from Pytorch_MobileNetV2 import MobileNetV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os


BEST_ACC = 0

def train_pytorch(checkpoint, pretrained=None):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pretrained:
        net = torch.load(pretrained).to(device)
    else:
        net = MobileNetV2(n_class=10, input_size=32).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        print("Testing...")
        test_pytorch(net, checkpoint, testloader)
        
def test_pytorch(net, checkpoint, testloader):
    global BEST_ACC

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    current_acc = np.sum(class_correct) / np.sum(class_total)
    if current_acc > BEST_ACC:
        BEST_ACC = current_acc
        torch.save(net, os.path.join(checkpoint, "best.pth"))
    else:
        torch.save(net, os.path.join(checkpoint, "last.pth"))

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    print("Overall accuracy: {}", current_acc)

class ImageNet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images_list = os.listdir(self.root_dir)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        self.categories = [item.replace(' ', '_') for item in categories]

        print("checking label...")
        print("found {} labels".format(self.__len__()))
        for idx in range(self.__len__()):
            label_name = '_'.join(self.images_list[idx].split('.')[0].split('_')[1:])
            if label_name in self.categories:
                pass
            else:
                print("{} not in categories".format(label_name))
        print("done checking!")

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images_list[idx])
        
        img = Image.open(img_path).convert('RGB')
        inp = self.transform(img)
        label_name = '_'.join(self.images_list[idx].split('.')[0].split('_')[1:])
        label = self.categories.index(label_name)

        return inp, label


def test_imagenet(net, imagenet_path, batch_size):
    dataset = ImageNet(imagenet_path)
    counter = 0
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8)
    with torch.no_grad():
        for inps, labels in tqdm(loader):  
            out = net(inps).numpy()
            out = np.argmax(out, axis=1)
            labels = labels.numpy()
            diff = out - labels
            counter += len(np.where(diff==0)[0])
        
    return counter/dataset.__len__()*100

if __name__=="__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="train mobilenet classifier")
    # parser.add_argument("--cp", help='checkpoint')
    # parser.add_argument("--pretrained", help='pretrain model')

    # args = parser.parse_args()

    # train_pytorch(args.cp, args.pretrained)
    net = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    net.eval()
    print(test_imagenet(net, "./imagenet-sample-images", batch_size=8))
