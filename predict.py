# PURPOSE: To successfully use the trained new network for classifying the images.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py --check_path <directory with the trained network (Required)> --image_path <path to the selected image> 
#      --top_k <number of probability class> --gpu <to select gpu or cpu for prediction, 0 = gpu, 1 = cpu> 
#      --cat_file <path to  JSON file that maps the class values to other category names>
#   Example call:
#   python predict.py --check_path .../data/checkpoint.pth --image_path .../flowers --top_k 5 --gpu 0 --cat_file .../data/cat_to_names.json

# Import all the necessary packages and modules 
import argparse
import numpy as np
import torch
import json
from PIL import Image
from torch import nn
import torch.nn.functional as F


# Main program function defined here
def main():
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()
    
    # Extract all supplied CL arguments and create variables
    check_path = in_arg.check_path
    image_path = in_arg.image_path
    top_k = in_arg.top_k
    cat_file = in_arg.cat_file
    gpu = in_arg.gpu
    
    # Setting the training device to Cuda if available else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu ==0 else "cpu")
    
    # Loading Checkpoint Dictionary
    checkpnt = load_checkpoint(check_path)
    arch = checkpnt['arch'] 
    
    # Creating Model from checkpoint 
    model, input_units = get_models(arch)
    
    # Extracting hyper parameters from checkpoint 
    hidden_units = checkpnt['hidden_units']
    output_units = checkpnt['output_units']
    epochs = checkpnt['epochs']
    
    # Defining Classifier
    classifier = net_classifier(arch, hidden_units, output_units, device,model, input_units)   
    
    # Freezing model parameters
    for param in model.parameters():
        param.requires_grad = False
     
    # Loading the model parameters
    if 'resnet' in arch:
        model.fc = classifier
    else:
        model.classifier = classifier
        
    model.load_state_dict(checkpnt['state_dict'])
    model.class_to_idx = checkpnt['class_to_idx']
    
    # Predict Top Probabilities for the selected Image file
    probs, classes = predict(image_path, model, gpu, top_k)
    print(probs)
    print(classes)

    mappings = category_to_name( probs, classes, cat_file)
    print("\nThe top {} predictions for image {} are : ".format(top_k,image_path))
    
    #for i, (a, b) in enumerate(zip(alist, blist)):
    for i, (cat, prob, class_id) in enumerate(zip(mappings, probs, classes), 1):
        print("{}.  {} : in Class: {} and with a probability of {:.2f}%\n".format(i, cat.title(),class_id, prob*100))
    
def load_checkpoint(filepath):
    """
    To load the saved checkpoint file.
    Parameters:
     Filepath 
    Returns:
     checkpoint 
    """
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    return checkpoint
    

def process_image(image_loc):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
        Parameters:
          image_loc 
        Returns:
          final_image 
    '''
    # Processing the target image resolution
    target = 224
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = Image.open(image_loc)
    
    # Resizing the image to 256 
    width, height = image.size   
    ratio = int(width)/int(height)
    
    if width <= height:
        image = image.resize((256, int(256/ratio)))
    else:
        image = image.resize((int(256/ratio), 256))
   
    # Cropping the image
    width, height = image.size   
    left = (width - target)/2
    top = (height - target)/2
    right = (width + target)/2
    bottom = (height + target)/2
    image = image.crop((left, top, right, bottom))
    
    # Normlizing
    np_image = np.array(image)
    image_norm = (np_image/255 - mean) / std
    
    # Transpose
    final_image = image_norm.transpose((2, 0, 1))
    
    return final_image

def predict(image_path, model, gpu, topk):
    ''' Returns the top 'k' predictions for the selected image
        Parameters:
          image_path 
          model 
          device  
          top_k 
        Returns:
          probs 
          top_label 
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu ==0 else "cpu")
    
    # Importing the image as a Numpy Array
    img = process_image(image_path)
    
    # Converting from Numpy to Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    image = image_tensor.unsqueeze(0)
    
    # Transferring the model to CPU
    model, image = model.to(device), image.to(device)
    model.eval()
    model.requires_grad = False
    
    # Calculating probabilities
    prob = torch.exp(model.forward(image))
    top_probs, top_labels = prob.topk(topk)
    
    probs, labels = top_probs.data.cpu().numpy()[0], top_labels.data.cpu().numpy()[0] 
    idx_to_class = {key: value for value, key in model.class_to_idx.items()}
    top_label = [idx_to_class[labels[i]] for i in range(labels.size)]
    
    return probs, top_label
    
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
    
    # Creates 5 command line arguments 
    # Use argparse Expected Call with <> indicating expected user input:
    #      python predict.py --check_path <directory with the trained network (Required)> --image_path <path to the selected image> 
    #      --top_k <number of probability class> --gpu <to select gpu or cpu for prediction, 0 = gpu, 1 = cpu> 
    #      --cat_file <path to  JSON file that maps the class values to other category names>    
    parser.add_argument('--check_path', '-p', type=str, default='data/checkpoint.pth', required=True,
                        help='Full path to directory containing saved checkpoint file ')
    parser.add_argument('--image_path','-i',type=str, default='D:/flowers/test/30/image_03482.jpg', help='full file path to single image to be predicted')
    parser.add_argument('--top_k', '-t', type=int, default=1,help='Number of probability classes')
    parser.add_argument('--gpu', '-d', type=int, default=-1,
                        help='To select gpu or cpu for prediction, 0 = gpu, 1 = cpu')
    parser.add_argument('--cat_file', '-f', type=str, default= 'cat_to_name.json',
                        help='path to  JSON file that maps the class values to other category names')
                        
    # returns parsed argument collection
    return parser.parse_args()


def net_classifier(arch, hidden_units, output_units, device, model, input_units):
    
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
    
    dropout = 1.0
    # model,input_units = get_models(arch)
    model.to(device)
    print("Loading Model Using {}".format(device))
    
    # Defining Classifier
    class Network(nn.Module):
        def __init__(self, input_size,output_size, hidden_layers, dropout):
            super().__init__()
            self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
            layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
            self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
            self.output = nn.Linear(hidden_layers[-1], output_size)
            self.dropout = nn.Dropout(dropout)
        # Forward pass function
        def forward(self, x):
            for each in self.hidden_layers:
                x = F.relu(each(x))
                #x = self.dropout(x)
            x = self.output(x)
            return F.log_softmax(x, dim=1)
    
    # Defining the classifier function
    classifier = Network(input_units, output_units, hidden_units, dropout)
    
    return classifier 
        
def get_models(arch):
    
    """
    Pretrained CNN whose architecture is indicated by this parameter, values must be: vgg and densenet models only (string)
    Parameters:
     arch 
    Returns:
     model, input_units 
    """
    
    from torchvision import datasets, transforms, models
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

def category_to_name(probs, classes, cat_file):
    """
    Function to allows users to load a JSON file that maps the class values to other category names
    Parameters:
     cat_file 
    Returns:
     cat_labels 
    """
    with open(cat_file, 'r') as f:
        cat_to_name = json.load(f)
    cat_labels = [cat_to_name[i] for i in classes]
    return cat_labels
    
# Call to main function to run the program 
if __name__ == '__main__':
    main()