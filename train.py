# Imports here
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import PIL
from PIL import Image
import argparse
from collections import OrderedDict
from torch.autograd import Variable

#Set the paths
data_dir = './assets/flower_data'

train_dir = data_dir + '/train'
test_dir = data_dir + '/test'
valid_dir = data_dir + '/valid'

#Define the transforms (Do for Training, Validation and Testing) to normalize the data
#Training Tranforms
data_transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
])


#Test Transforms
data_transforms_test = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
])

#Validation data set
data_transforms_validation = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])
])

#Load the datasets with ImageFolder
image_datasets_train = datasets.ImageFolder(train_dir, transform=data_transforms_train)
image_datasets_test = datasets.ImageFolder(test_dir, transform=data_transforms_test)
image_datasets_validation = datasets.ImageFolder(valid_dir, transform=data_transforms_validation)

#Using the image datasets and the trainforms, define the dataloaders
dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=32, shuffle=True)
dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, batch_size=32, shuffle=True)
dataloaders_validation = torch.utils.data.DataLoader(image_datasets_validation, batch_size=32, shuffle=True)


#label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
####Network build begins here###

#CLarify details for user
print("Choose between three models, select the learning rate and determine if you would like to use gpu")
print("You have three options for the models, vgg16, densenet121 and resnet50. Please use all smaller letters")

#Choose the model
chosen_model = input("Which model would you like to use? ")

#Select the learning rate
learning_rate = float(input("Enter learning rate: for example 0.003: "))


#Build a feed forward network
def Feed_Forward_Network(chosen_model, learning_rate):
    if (chosen_model == 'resnet50'):
        model = models.resnet50(pretrained=True)
        print("Lets get training this puppy with the resnet50 model!")
    elif (chosen_model == 'densenet121'):
        model = models.densenet121(pretrained=True)
        print("Lets get training this puppy with the densenet121 model!")
    elif (chosen_model == 'vgg16'):
        model = models.vgg16(pretrained=True)
        print("Lets get training this puppy with the vgg16 model!")
        
    if (learning_rate > 0.01):
        print("Woah easy there tiger! Lets go for something a little bit lower")
        print("Try again")
        learning_rate = float(input("Enter learning rate: for example 0.003: "))
    else:
        print("Great thanks!")
    
    #Freeze the parameters
    for param in model.parameters():
        param.requires_grad = True
    
    #Define the new classfier
    model.classifier = nn.Sequential(nn.Linear(2048, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 102),
                                 nn.LogSoftmax(dim=1))
    model.fc = model.classifier
    
    #Set the loss
    criterion = nn.NLLLoss()
    
    #Run optimizer
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    
    
    #Determine GPU or not
    cuda_chosen = input("Would you like to use gpu: ")
    if (cuda_chosen == 'Yes'):
        model.to('cuda')
    
    print(optimizer)
    
    return model , optimizer ,criterion
    

#Call the function
model,optimizer,criterion = Feed_Forward_Network(chosen_model, learning_rate)

print(model.fc)


#Train the model

#Set the standard variables
epochs = 3
steps = 0
running_loss = 0
print_every = 5

#Run through the loops
for epoch in range(epochs):
    for images, labels in dataloaders_train:
        steps += 1
        images, labels = images.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        #Forward and backwards pass
        log_probabilities = model(images)
        loss = criterion(log_probabilities, labels)
        loss.backward()
        optimizer.step()
        
        #Tally up running loss
        running_loss += loss.item()
        
        #Drop out and validate the set
        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0
            
            for images, labels in dataloaders_test:
            
                images, labels = images.to('cuda'), labels.to('cuda')

                log_probabilities = model(images)
                loss = criterion(log_probabilities, labels)
                test_loss += loss.item()

                #Calculate the accuacy
                ps = torch.exp(log_probabilities)
                top_ps, top_class = ps.topk(1, dim=1)
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor))
                
            print(f"Epoch: {epoch+1}/{epochs}: " f"Training Loss: {running_loss/print_every:.3f}.."
                 f"Test Loss: {test_loss/len(dataloaders_test):.3f}.."
                 f"Test Accuracy: {accuracy/len(dataloaders_test):.3f}")

            #Set running loss and model back to train
            running_loss = 0
            model.train()
            
            
            
# TODO: Do validation on test set

#Function to test the dataloaders_test set

def calc_test_set_accuracy(dataloaders_test):
    epochs = 1
    steps = 0
    running_loss = 0
    print_every = 5
    
    for epoch in range(epochs):
        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0

            for images, labels in dataloaders_validation:

                images, labels = images.to('cuda'), labels.to('cuda')

                log_probabilities = model(images)
                loss = criterion(log_probabilities, labels)
                test_loss += loss.item()

                #Calculate the accuacy
                ps = torch.exp(log_probabilities)
                top_ps, top_class = ps.topk(1, dim=1)
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor))

            print(f"Epoch: {epoch+1}/{epochs}: " f"Training Loss: {running_loss/print_every:.3f}.."
                 f"Test Loss: {test_loss/len(dataloaders_test):.3f}.."
                 f"Test Accuracy: {accuracy/len(dataloaders_test):.3f}")

            #Set running loss and model back to train
            running_loss = 0
            model.train()

#Call the function
calc_test_set_accuracy(dataloaders_test)




# TODO: Save the checkpoint

#Create check point dict
checkpoint = {
    'chosen_model' : model,
    'input_size': 2048,
    'output_size': 102,
    'dropout' : 0.2,
    'state_dict' : model.state_dict()
}

torch.save(checkpoint, 'checkpoint.pth')
print(model.state_dict())


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = nn.Sequential(checkpoint['input_size'],
                    checkpoint['output_size'],
                    checkpoint['dropout'])
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

model = load_checkpoint('checkpoint.pth')




def process_image(image):
    #Open up the image
    img = Image.open(image)
    
    
    #Show image
    print("Initial Image Size")
    print(img.size)
    
    #Resize and transform image
    image = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
    ])
    
    #Image Tensor    
    print("Image Tensor Below")
    print(image)
    
    final_image = image(img)
    
    return final_image
    
    
#Call the function
selected_image = test_dir + '/1/' + 'image_06760.jpg'
print("Image Path is " + selected_image)
image_for_imshow = process_image(selected_image)



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# TODO: Display an image along with the top 5 classes
def view_graphs():
    #Set the x and y variables for the chart
    flowers = flower_names
    y_pos = np.arange(len(flowers))
    performance = probs
    
    plt.bar(y_pos, performance)
    plt.xticks(y_pos, flowers)
    plt.ylabel('FLower Accuracy')
    plt.title("Bar Graph to display flower accuracy")

    plt.show()
    
#Call function
view_graphs()
imshow(process_image(selected_image))