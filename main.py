from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune
import numpy as np
import os

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

        # print("checking label...")
        # print("found {} labels".format(self.__len__()))
        # for idx in range(self.__len__()):
        #     label_name = '_'.join(self.images_list[idx].split('.')[0].split('_')[1:])
        #     if label_name in self.categories:
        #         pass
        #     else:
        #         print("{} not in categories".format(label_name))
        # print("done checking!")

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images_list[idx])
        
        img = Image.open(img_path).convert('RGB')
        inp = self.transform(img)
        label_name = '_'.join(self.images_list[idx].split('.')[0].split('_')[1:])
        label = self.categories.index(label_name)

        return inp, label

def init_net():
    net = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    net.eval()

    return net

def test_imagenet(net, imagenet_path, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    dataset = ImageNet(imagenet_path)
    counter = 0
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8)

    with torch.no_grad():
        for inps, labels in tqdm(loader):  
            inps = inps.to(device)
            out = net(inps).cpu().numpy()
            out = np.argmax(out, axis=1)

            labels = labels.cpu().numpy()
            diff = out - labels
            counter += len(np.where(diff==0)[0])
        
    return counter/dataset.__len__()*100

def prune_random_unstructured(net, imagenet_path, batch_size):
    for i in range(1,5):
        for idx in range(2,18):
            module = net.features[idx].conv
            amount = i/10
            prune.random_unstructured(module[0][0], name='weight', amount=amount)
            prune.random_unstructured(module[1][0], name='weight', amount=amount)
            result = test_imagenet(net, imagenet_path, batch_size)

            with open("log.txt",'a+') as file:
                file.write("method: rand_unstr - module: {} - prune amount: {:.0%} - accuracy: {} \n".format(idx, amount, result))


def prune_global_unstructured(net, imagenet_path, batch_size):
    for i in range(1,10):
        amount = i/10
        module = net.features
        para_to_prune = []
        for idx in range(2,18):
            sub_module = module[idx]
            conv2d_1 = sub_module.conv[0][0]
            conv2d_2 = sub_module.conv[1][0]
            para_to_prune.append((conv2d_1, 'weight'))
            para_to_prune.append((conv2d_2, 'weight'))

        prune.global_unstructured(
            para_to_prune,
            pruning_method=prune.L1Unstructured,
            amount = amount
        )

        result = test_imagenet(net, imagenet_path, batch_size)
        with open("log.txt",'a+') as file:
            file.write("method: glob_unstr - prune amount: {:.0%} - accuracy: {} \n".format(amount, result))

if __name__=="__main__":
    net = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    net.eval()
    # print(test_imagenet(net, "./imagenet-sample-images", batch_size=8))
    # torch.save(net, "prune_model/base.pth")
    prune_global_unstructured(net, "./imagenet-sample-images", batch_size=8)
    prune_random_unstructured(net, "./imagenet-sample-images", batch_size=8)
