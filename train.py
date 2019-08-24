import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json

print(torch.__version__)

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

#Load the pre trained network
model = models.resnet50(pretrained=True)

#Print out the model
print("Model details below")
print(model)

#Turn off the gradients for the model
for param in model.parameters():
    param.requires_grad = False

#Define the new feew forward classifier
classifier = nn.Sequential(nn.Linear(4096, 512),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(512, 2),
                            nn.LogSoftmax(dim=1))

model.fc = classifier

#Print amended model
print("New Model details below")
print(model.fc)

#Determine the loss criterion
criterion = nn.NLLLoss()

optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

#####Traing the model#####
#Setting the variables
epochs = 1
steps = 0
running_loss = 0
print_every = 5

#Run through the loop for the training data
for epoch in range(epochs):
    for images, labels in dataloaders_train:
        steps += 1

        #images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        log_probabilities = model(images)
        loss = criterion(log_probabilities, labels)
        #loss.backward()
        optimizer.step()

        #Calc running loss
        running_loss += loss.item()

        #Drop out of the trainig loop and test the accuracy on test data
        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0

            for images, labels in dataloaders_test:

                log_probabilities = model(images)
                loss = criterion(log_probabilities, labels)
                test_loss += loss.item()

                #Calc the accuracy
                ps = torch.exp(log_probabilities)
                top_ps, top_class = ps.topk(1, dim=1)
                
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(accuracy.type(torch.FloatTensor()))

            #Print the data
            print("Epoch data {epoch+1}/{epochs}")

            running_loss = 0
            model.train()




#Check point data
checkpoint = {
    'input_size' : 4096,
    'output_size' : 512,
    'hidden_layers' : [each.out_features for each in model.hidden_layers],
    'state_dict' : model.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')

#Load the checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = model.(checkpoint['input_size'],
                        checkpoint['output_size'],
                        checkpoint['hidden_layers'],
                        checkpoint['state_dict'])

    model.load_state_dict(state_dict)

    return model