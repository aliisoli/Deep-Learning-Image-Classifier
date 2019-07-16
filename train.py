
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import argparse , torch
import numpy as np , pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim


def get_args():
    # default variables are set for speed, not accuracy of the mode. highly recommended to increase epochs
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", type=str, default = 'flowers')
    parser.add_argument("--save_dir", type=str, default="checkpoint_Ali.pth")
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden_units", type=list, default=[700, 300])
    parser.add_argument("--output", type=int, default=102)
    parser.add_argument("--gpu", type=bool, default=True)
    parser.add_argument("--arch", type=str, default="vgg")
    
    return parser.parse_args()




class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop):
        super().__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        i = 0
        j = len(hidden_layers)-1
        
        while i != j:
            l = [hidden_layers[i], hidden_layers[i+1]]
            self.hidden_layers.append(nn.Linear(l[0], l[1]))
            i+=1
        for each in hidden_layers:
            print(each)

        self.output = nn.Linear(hidden_layers[j], output_size)
        self.dropout = nn.Dropout(p = drop)
 
    def forward(self, tensor):
        for linear in self.hidden_layers:
            tensor = F.relu(linear(tensor))
            tensor = self.dropout(tensor)
        tensor = self.output(tensor)

        return F.log_softmax(tensor, dim=1)

    
    
def main():
    
    
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    
    input = get_args()
    
    data_dir = input.data_directory
    save_to = input.save_dir
    learning_rate = input.learning_rate
    epochs = input.epochs
    hidden_layers = input.hidden_units
    output_size = input.output
    dropout = 0.4
    pretrained_model = input.arch
    gpu = input.gpu
    

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
 
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])


   
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle = True)
    
    #model = models.vgg16(pretrained=True)
    #input_size = 25088
    # changed to multiple archs
    
    model_dict = {"vgg": vgg16, "resnet": resnet18, "alexnet": alexnet}
    inputsize_dict = {"vgg": 25088, "resnet": 512, "alexnet": 9216}
    
    model = model_dict[pretrained_model]
    input_size = inputsize_dict[pretrained_model]
    
    if gpu==True:
        if torch.cuda.is_available():
            device = 'cuda'
            print('Using GPU for calculations')
        else:
            device = 'cpu'
            print('Your system is not compatible with CUDA')
            print('Using CPU for calculations')
    else:
      device = 'cpu'
      print('Using CPU for calculations')
    
    model.to(device)
    print('device: ', device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    
    classifier = NeuralNetwork(input_size, output_size, hidden_layers, dropout)
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    #####################     Train model      #######################################
    
    print("The Neural Network is being trained... Be patient")
    valid_len = len(validloader)
    print_every = 32
    steps = 0
    # change to cuda
    #device = torch.device('cuda')
    
    epoch = 0
    while epoch < epochs:
        running_loss = 0
        
        for inputs, labels in iter(trainloader):
            if gpu==True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                #print('transferring images and labels to cuda')
                model = model.cuda()
            steps += 1
            #inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    
            if steps % print_every == 0:
                model.eval()
                val_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for images, labels in validloader:
                        if gpu==True:
                            #print('transferring images and labels to cuda')
                            images, labels = images.to('cuda'), labels.to('cuda')
                            model = model.cuda()
                        #images, labels = images.to(device), labels.to(device)

                        #forward
                        
                        logps = model.forward(images)
                        val_loss += criterion(logps, labels).item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1,dim=1)

                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"valid loss: {val_loss/len(validloader):.3f}.. "
                          f"validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
            
        
        epoch +=1

    print('training is done, let us see the accuracy of the model on testing set')
    
    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if gpu==True:
                images, labels = images.to('cuda'), labels.to('cuda')
                model = model.cuda()
            #images, labels = images.to(device), labels.to(device)
            
            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            prob = torch.exp(output)
            pred = prob.max(dim=1) 

            
            matches = (pred[1] == labels.data)
            correct += matches.sum().item()
            total += 64

        acc = 100*(correct/total)
    print("Test Accuracy is: ", acc)
    
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_layers,
                  'dropout': dropout,
                  'epochs': epochs,
                  'learning_rate': learning_rate,
                  'optimizer': optimizer,
                  'classifier' : model.classifier,
                  'state_dict': model.state_dict(),
                  'arch': pretrained_model,
                  'class_to_idx' : model.class_to_idx}

    torch.save(checkpoint, save_to)


if __name__ == "__main__":
    main()
