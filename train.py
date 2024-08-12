# PURPOSE: To successfully train a new network on a dataset of images and saves the model to a checkpoint.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py --data_dir <directory with images> --epochs <number of epochs> --arch <model>
#             --learning_rate <alpha> --gpu <to select gpu or cpu for training, 0 = gpu, 1 = cpu> 
#             --save_dir <save directory> --hidden_units <number of hidden units>
#   Example call:
#   python train.py --data_dir .../flowers --epochs 4 --arch vgg16 --learning_rate 0.0001 --gpu 0 --save_dir data --hidden_units 4096

# Importing all the necessary packages and modules 
import os
import torch
import time
import argparse
import json
import numpy as np
from torch import optim
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from time import time, sleep
import torch.nn.functional as F

#Defining Paramaters
dropout = 0.5
output_units = 102
batch_size = 64
test_batch_size = 32

# Main program function defined below
def main():
    # Measures total program runtime by collecting start time
    start_time = time()
    
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()
    
    #Extract all supplied CL arguments and create variables
    data_dir = in_arg.data_dir
    epochs = in_arg.epochs
    arch = in_arg.arch
    hidden_units = in_arg.hidden_units
    learning_rate = in_arg.learning_rate
    gpu = in_arg.gpu
    save_dir = in_arg.save_dir
    
    # Setting the training device to Cuda if available else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu ==0 else "cpu")
    
    # Loading the training , validation and test data 
    loaders = preprocess(data_dir, batch_size, test_batch_size)
    
    # Building and training the classifier
    model,input_units = get_models(arch)

    # Freezing Model parameters 
    for param in model.parameters():
        param.requires_grad = False
    
    # Defining Classifier
    classifier = net_classifier(arch, hidden_units, output_units, dropout, device, model, input_units)

    if 'resnet' in arch:
        model.fc = classifier
        # Defining Criterion & Optimizers 
        criterion = nn.NLLLoss()
        criterion = criterion.to(device)
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
        # Training the Network
        train = train_model(model, criterion, optimizer, epochs, device,loaders, save_dir,model.fc)

    else:
        model.classifier = classifier
        # Defining Criterion & Optimizers 
        criterion = nn.NLLLoss()
        criterion = criterion.to(device)
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        # Training the Network
        train = train_model(model, criterion, optimizer, epochs, device,loaders, save_dir,model.classifier)

    op_dict = train[0]
    mod_dict = train[1]
    dataloader = loaders[0]['train_data']
    
    train_data = dataloader
    model.class_to_idx = train_data.class_to_idx
    model.to(device)

    checkpoint = {'output_units': output_units,
                  'hidden_units': hidden_units,
                  'epochs': epochs,
                  'learning_rate': learning_rate,
                  'arch' : arch,
                  'batch_size': batch_size,
                  'class_to_idx': model.class_to_idx,                    
                  'state_dict' : mod_dict}
    
    # Saving Checkpoint
    filepath  = save_dir+"/"+"checkpoint.pth"

    if save_dir:
        if os.path.exists(save_dir):
            torch.save(checkpoint,filepath)  
            print("Checkpoint Saved: '{}'".format(filepath))
        else:
            os.makedirs(save_dir)
            torch.save(checkpoint,filepath)
            print("Checkpoint Saved: '{}'".format(filepath))
    # Calculating program runtime
    end_time = time()
    
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

  
# Functions defined below
def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates parse 
    parser = argparse.ArgumentParser(description='Train the Network using Transfer Learning to Classify Flowers')
    
    # Creates 7 command line arguments 
    # --data_dir <directory with images> --epochs <number of epochs> --arch <model>
    #             --learning_rate <alpha> --gpu <to select gpu or cpu for training, 0 = gpu, 1 = cpu> 
    #             --save_dir <save directory> --hidden_units <number of hidden units>
    
    parser.add_argument('--data_dir', '-d', type=str, default='d:/flowers', required=True,
                        help='Location of the images to be utilized')
    parser.add_argument('--arch', '-m',type=str, default='vgg16', 
                        help='Model to be used for training')  
    parser.add_argument('--save_dir','-s',type=str, default='data', help='Location to save the trained model')
    parser.add_argument('--learning_rate', '-l', type=float, default=0.0001, 
                        help='Learning rate of the model')
    parser.add_argument('--epochs', '-e', type=int, default=5,
                        help='Number of epochs to be used during the training process')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='Selection between GPU (0 = GPU) and CPU (1 = CPU)')
    parser.add_argument('--hidden_units', '-u', nargs='+', type=int, default=[4096, 512],
                        help='Number of hidden units')
                        
    # returns parsed argument collection
    return parser.parse_args()

def preprocess(dir, batch_size, test_batch_size):
    """
    Receives the Image Directory as input and applies torchvision transforms to augment the training data with random     scaling, rotations, mirroring.
    Also applies transforms and normalisations to the testing and validation dat
    
    Parameters:
     dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     image_datasets, Dataloaders - Dictionary containing the Image and Dataloader dictionaries  
    """
    # Load the training , validation and test data and apply transforms and normalization to the images
    data_dir = dir
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.Resize(256),
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],  
                                                            [0.229, 0.224, 0.225])])


    test_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])])
 
    data_transforms = {'train':train_transforms,'valid':validation_transforms,'test':test_transforms}
 
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    validation_data = datasets.ImageFolder(test_dir, transform=validation_transforms)

    image_datasets = {'train_data':train_data,'test_data':test_data,'validation_data':validation_data}
   
    # Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, test_batch_size)
    validloader = torch.utils.data.DataLoader(validation_data, test_batch_size)

    dataloaders = {'trainloader':trainloader,'testloader':testloader,'validloader':validloader}
    # Function returns Image Datasets & Dataloaders Dictionaries 
    return image_datasets, dataloaders

def get_models(arch):
    """
    Pretrained CNN whose architecture is indicated by this parameter, values must be: vgg and densenet models only (string)
    Parameters:
     arch 
    Returns:
     model, input_units 
    """
    if 'vgg' in arch:
        valid = [11,13,16,19]
        arch_units =  ''.join(i for i in arch if i.isdigit())
        if int(arch_units) in valid: 
            model = "models"+"."+arch+"(pretrained=True)"
            model = eval(model)
            input_units = model.classifier[0].in_features
        else:
            print('Unrecognized Model!!! Please pass a valid Model')
    elif 'densenet' in arch:
        valid = [121,169,201,161]
        arch_units =  ''.join(i for i in arch if i.isdigit())
        if int(arch_units) in valid: 
            model = "models"+"."+arch+"(pretrained=True)"
            model = eval(model)
            input_units = model.classifier.in_features
        else:
            print('Unrecognized Model!!! Please pass a valid Model')    
    elif 'resnet' in arch:
        valid = [18,34,50,101,152]
        arch_units =  ''.join(i for i in arch if i.isdigit())
        if int(arch_units) in valid: 
            model = "models"+"."+arch+"(pretrained=True)"
            model = eval(model)
            input_units = model.fc.in_features
        else:
            print('Unrecognized Model!!! Please pass a valid Model')    
    else:
        print('Unexpected network architecture', model)
    
    return model, input_units

    
def net_classifier(arch, hidden_units, output_units, dropout, device,model, input_units):
    """
    A new feedforward network is defined for use as a classifier using the features as inputs.
    
    Parameters:
    arch - Pretrained CNN model as input - e,g VGG13
    hidden_units - No of hidden units
    output_units - number of output units
    dropout - dropout ratio 
    gpu - GPU State 
    device - Device State - GPU or CPU 
    Returns:
    model.classifier, model - Returns the model and model.classifier
    
    """
    # model,input_units = get_models(arch)
    model.to(device)
    print("Training Network using :{}\n".format(device))
    print("This would take a couple of minutes")
    
    # Define Classifier
    class Network(nn.Module):
        def __init__(self, input_size,output_size, hidden_layers, drop_p=0.50):
            super().__init__()
            self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
            layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
            self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
            self.output = nn.Linear(hidden_layers[-1], output_size)
            self.dropout = nn.Dropout(drop_p)
       
        # Forward pass
        def forward(self, x):
            for each in self.hidden_layers:
                x = F.relu(each(x))
                x = self.dropout(x)
            x = self.output(x)
            return F.log_softmax(x, dim=1)
     
    # Defining the classifier function
    classifier = Network(input_units, output_units, hidden_units, drop_p = 0.5)
    
    return classifier
        
def train_model(model, criterion, optimizer, epochs, device,loader,store_dir, classifier):
    """
    This function trains the parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static
    
    Parameters:
    arch - Pretrained CNN model as input - e,g VGG13
    hidden_units - No of hidden units
    output_units - number of output units
    dropout - dropout ratio 
    gpu - GPU State 
    device - Device State - GPU or CPU 
    Returns:
    model.classifier, model - Returns the model and model.classifier
    
    """
    total_steps = len(loader[1]['trainloader'])
    print_every = 40
    steps = 0
    trainloader = loader[1]['trainloader']
    validloader = loader[1]['validloader']
    
    # change to device
    model.to(device)
    begin = time()
    
    for e in range(epochs):
        model.train()
        running_loss = 0
        start = time()
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            # inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
           
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                
                accuracy = 0
                valid_loss = 0
                with torch.no_grad():
                    for images, labels in validloader:
                        images = images.to(device)
                        labels = labels.to(device)
                
                        output = model.forward(images)
                        valid_loss += criterion(output, labels).item()
                    
                        # Calculate the Accuracy 
                        ps = torch.exp(output).data
                        equality = (labels.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                        
                    print("Epoch: {}/{}... ".format(e+1, epochs), 
                          "| Training Loss: {:.3f}..... ".format(running_loss/print_every),
                          "| Validation Loss: {:.3f}..... ".format(valid_loss/len(validloader)),
                          "| Accuracy: {:.3f}".format(accuracy/len(validloader))) 
                   
                    running_loss = 0
                    model.train()       
        print(f"Device = {device}; Time per batch: {(time() - start)/3:.3f} seconds")
    print(f"Total Time: {(time() - begin):.3f}")  
        
    op_dict = optimizer.state_dict()
    mod_dict = model.state_dict()
    
    return op_dict, mod_dict

# Call to main function to run the program
if __name__ == '__main__':
    main()