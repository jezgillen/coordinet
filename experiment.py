#!/usr/bin/python3

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import json
import datetime
#  import interrupt
import matplotlib.pyplot as plt

class FullyConnected(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        ''' arguments: int, int, list<int> '''
        super().__init__()
        # declare parameters
        self.w = nn.ModuleList() # list of weight matricies

        # this loop initialises all weights except last layer
        for h in hidden_sizes:
            self.w.append(nn.Linear(input_size, h))
            input_size = h

        self.w.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        batch = x.shape[0]
        x = torch.reshape(x,(batch, -1))
        # define computation
        for w in self.w[:-1]:
            x = F.relu(w(x))

        return self.w[-1](x) # this outputs the logits

    def predict(self, x):
        return F.softmax(self.forward(x))

    def num_parameters(self):
        num_parameters = 0
        for p in self.parameters():
            num_parameters += np.prod(p.shape)
        return num_parameters

class Coordinet(nn.Module):
    def __init__(self, input_size, embedding_size, output_size, embed_hidden_sizes,decode_hidden_sizes):
        ''' arguments: int, int, list<int> '''
        super().__init__()
        # declare parameters
        self.w = nn.ModuleList() # list of weight matricies

        # this loop initialises all weights except last layer
        for h in embed_hidden_sizes:
            self.w.append(nn.Linear(input_size, h))
            input_size = h

        self.w.append(nn.Linear(input_size, embedding_size))

        self.embed_to_output = FullyConnected(embedding_size, output_size, decode_hidden_sizes)
        self.w.append(self.embed_to_output)

    def coordinate_encoder(self, x):
        ''' takes image x of shape [batch, d, h, w] and maps to [batch, d+2, h*w] '''
        shape = x.shape
        # create coordinates of shape [batch, h, w, 2]
        h = torch.arange(0,shape[2])
        w = torch.arange(0,shape[3])
        h = torch.transpose(torch.tile(h,(shape[0],1,shape[3],1)),3,2)
        w = torch.tile(w,(shape[0],1,shape[2],1))
        coords = torch.cat([h,w],dim=1)
        if GPU:
            coords = coords.cuda()
        coords = coords/28
        x = torch.cat([x,coords],dim=1)
        x = torch.reshape(x, (shape[0], x.shape[1], shape[2]*shape[3]))
        x = torch.transpose(x,1,2)
        return x

    def process_pixels(self, x):
        # define computation
        for w in self.w[:-2]:
            x = F.relu(w(x))

        return self.w[-2](x) # this outputs the logits

    def forward(self, x):
        x = self.coordinate_encoder(x)

        # (pixel -> embedding) module
        # [batch, h*w, d] -> [batch, h*w, embedding]
        batch, hxw, depth = x.shape
        x = torch.reshape(x, (-1, depth))
        embeddings = self.process_pixels(x)
        embeddings = torch.reshape(embeddings,(batch,hxw,-1))
        embedding = torch.sum(embeddings, dim=1)

        # (embedding -> output) module
        # [batch, embedding] -> [batch, output]
        output = self.embed_to_output(embedding)

        return output

    def predict(self, x):
        return F.softmax(self.forward(x))

    def num_parameters(self):
        num_parameters = 0
        for p in self.parameters():
            num_parameters += np.prod(p.shape)
        return num_parameters


def training_loop(model, train_dataset, test_dataset):

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=BATCH_SIZE, 
                                              shuffle=False)

    # set up for training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,)# momentum=momentum)
    loss_criterion = nn.CrossEntropyLoss()


    # Data dict
    training_record = dict(epoch=[], 
                           training_acc=[], 
                           validation_acc=[], 
                           training_loss=[], 
                           validation_loss=[])

    # training loop
    curr_loss = 10e8
    epoch = 0
    while(epoch < NUM_EPOCHS and curr_loss > LOSS_THRESHOLD):
        total_correct = 0
        curr_loss = 0
        num_training_images = 0
        for i, (X, y) in enumerate(train_loader):
            if GPU:
                X = X.cuda()
                y = y.cuda()
            
            #  Forward pass
            outputs = model(X)
            prediction = torch.argmax(outputs, dim=1)
            loss = loss_criterion(outputs, y)
            curr_loss += loss.item()*y.shape[0] #curr batch size
            total_correct += torch.sum(prediction == y)
            num_training_images += y.shape[0]
            
            # Backprpagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        training_record['epoch'].append(epoch)
        training_record['training_loss'].append(float(curr_loss/num_training_images))
        training_record['training_acc'].append(float(total_correct/num_training_images))

        # run test
        with torch.no_grad():
            total_correct = 0
            total_loss = 0
            total_images = 0
            for X, y in test_loader:
                if GPU:
                    X = X.cuda()
                    y = y.cuda()
                outputs = model(X)

                prediction = torch.argmax(outputs, dim=1) # max outputs max and argmax
                num_correct = torch.sum(prediction == y)
                total_correct += num_correct.item()
                total_images += y.shape[0]

                total_loss += loss_criterion(outputs, y).item()*y.shape[0]

            training_record['validation_acc'].append(float(total_correct / total_images))
            training_record['validation_loss'].append(float(total_loss / total_images))

        print(f"Epoch {epoch}, Training Acc: {training_record['training_acc'][epoch]:.3f}, Validation acc: {training_record['validation_acc'][epoch]:.3f}")
        epoch += 1
    print()

    return training_record

###########################
# Get dataset

def flatten(x):
    return torch.reshape(x,(-1,))

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #  flatten,
    ])

train1 = torchvision.datasets.FashionMNIST('./data/',train=True,transform=transform, download=True)
test1 = torchvision.datasets.FashionMNIST('./data/',train=False,transform=transform, download=True)

###########################
# Define Hyperparameters

BATCH_SIZE = 256
NUM_EPOCHS = 50
LR = 0.0001
MOMENTUM = 0.9
LOSS_THRESHOLD = 0.0
BREAK_TRAINING = False
GPU = torch.cuda.is_available()

##############################
# Run test

#  model = Coordinet(3, 30, 10, [],[20])
model = Coordinet(3, 200, 10, [200,200],[100,80])
if GPU:
    model = model.cuda()
stats1 = training_loop(model,train1,test1)


############################
# Save results (can be turned into graphs later)

timestamp = datetime.datetime.now().strftime("%M-%H-%d-%m")
experiment_name = "initial_test"
filename = f"{experiment_name}-{timestamp}.json"
json.dump(stats1, open(filename, 'w'))

try:
    from google.colab import files
    files.download(filename)
except:
    pass
