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
import seaborn as sb
import PIL
from PIL import Image
import argparse
from collections import OrderedDict
from torch.autograd import Variable
from train.py import train


#Used this blog to help me with the predict function https://medium.com/datadriveninvestor/creating-a-pytorch-image-classifier-da9db139ba80

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    image_to_predict = process_image(selected_image)
    
    image_to_predict = np.expand_dims(image_to_predict, 0)
    
    image_to_predict = torch.from_numpy(image_to_predict)
    
    model.eval()
    
    inputs = Variable(image_to_predict).to('cuda')
    logits = model.forward(inputs)
    
    ps = F.softmax(logits,dim=1)
    topk = ps.cpu().topk(topk)
    
    return (i.data.numpy().squeeze().tolist() for i in topk)


probs, classes = predict(selected_image, model)

#Print out the classes
print(probs)
print(classes)

#Class names for test data set
image_datasets_test_class = image_datasets_test.classes

flower_names = [cat_to_name[image_datasets_test_class[i]] for i in classes]
print(flower_names)