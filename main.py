from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
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
    net = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    net.eval()
    print(test_imagenet(net, "./imagenet-sample-images", batch_size=8))
