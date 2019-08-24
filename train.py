import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json

#Assign the data directories
data_dir = './assets/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Define the transforms (Do for Training, Validation and Testing) to normalize the data
#Training Tranforms
data_transforms_train = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225])
])


#Test Transforms
data_transforms_test = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225])
])

#Load the datasets with ImageFolder
image_datasets_test = datasets.ImageFolder(data_dir, transform=data_transforms_test)

image_datasets_train = datasets.ImageFolder(data_dir, transform=data_transforms_train)

#Using the image datasets and the trainforms, define the dataloaders
dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, batch_size=32, shuffle=True)

dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=32, shuffle=True)


#Map the labels
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#Use GPU if availble
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load the pre trained network
model = models.vgg11(pretrained=True)

print(model)
