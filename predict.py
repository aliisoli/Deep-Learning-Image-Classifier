import argparse, numpy as np , PIL
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

##    NOTE:
# At the end when I run the script, I get the following error: QXcbConnect

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, default="flowers/valid/1/image_06739.jpg")
    parser.add_argument("checkpoint", type=str, default="checkpoint_Ali.pth")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--category_names", type=str, default="cat_to_name.json")
    parser.add_argument("--gpu", type=bool, default=True)
    return parser.parse_args()


class NeuralNetwork(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        i = 0
        j = len(hidden_layers)-1
        while i != j:
            l = [hidden_layers[i], hidden_layers[i+1]]
            self.hidden_layers.append(nn.Linear(l[0], l[1]))
            i+=1
        self.output = nn.Linear(hidden_layers[j], output_size)    
    def forward(self, tensor):

        for linear in self.hidden_layers:
            tensor = F.relu(linear(tensor))
        tensor = self.output(tensor)
        return F.log_softmax(tensor, dim=1)
    
    

def load_checkpoint(path):
    checkpoint = torch.load(path)
    #model = models.vgg16(pretrained=True);
    #model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    pretrained_model = checkpoint['arch']
    model_dict = {"vgg": vgg16, "resnet": resnet18, "alexnet": alexnet}
    inputsize_dict = {"vgg": 25088, "resnet": 512, "alexnet": 9216}
    model = model_dict[pretrained_model]
    input_size = inputsize_dict[pretrained_model]
    
    for param in model.parameters(): param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    
    return model



def process_image(image):

    test_image = PIL.Image.open(image)
    width, height = test_image.size
    
    if width < height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]
        
    test_image.thumbnail(size=resize_size)

    center = width/4, height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))

    
    np_image = np.array(test_image)/255 

    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
        
    
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

def predict(image_path, model, topk ,gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu:
        loaded_model = load_checkpoint(model).cuda()
    # Pre-processing image
    img = process_image(image_path)
    # Converting to torch tensor from Numpy array
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img_tensor.unsqueeze_(0)

    # Setting model to evaluation mode and turning off gradients
    loaded_model.eval()
    with torch.no_grad():
        # Running image through network
        if gpu:
            img_add_dim = img_add_dim.to('cuda')
            loaded_model = loaded_model.to('cuda')
        output = loaded_model.forward(img_add_dim)

    # Calculating probabilities
    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    # Loading index and class mapping
    class_to_idx = loaded_model.class_to_idx
    # Inverting index-class dictionary
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    # Converting index list to class list
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
        
    return probs_top_list, classes_top_list

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
    
    
    
def main():
    
    input = get_args()
    
    path_to_image = input.image_path
    model_path = input.checkpoint
    num = input.top_k
    cat_names = input.category_names
    checkpoint = input.checkpoint
    gpu = input.gpu
    
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
        
    
    model = load_checkpoint(model_path)
    
    
    
    probs,classes = predict(path_to_image, checkpoint, num,gpu)


    # Converting classes to names
    names = []
    for i in classes:
        names += [cat_to_name[i]]
    # Getting prediction
    probs,classes = predict(path_to_image, checkpoint, num,gpu)

    # Converting classes to names
    names = []
    for i in classes:
        names += [cat_to_name[i]]

    # Creating PIL image
    image = PIL.Image.open(path_to_image)
    
    # Plotting test image and predicted probabilites
    f, ax = plt.subplots(2,figsize = (6,10))

    ax[0].imshow(image)
    ax[0].set_title(names[0])

    y_names = np.arange(len(names))
    ax[1].barh(y_names, probs, color='red')
    ax[1].set_yticks(y_names)
    ax[1].set_yticklabels(names)
    ax[1].invert_yaxis() 

    plt.show()
    
    
    
    
    


if __name__ == "__main__":
    main()