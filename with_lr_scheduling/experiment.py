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


# TODO: Make multiplicative version instead of concat to depth?
#       Make additive one 
#       Learned positional vectors?
class PixelCoordinateEmbeddings(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.w = nn.ModuleList()

        sin_embedding_size = embedding_size-2-1
        self.B = nn.Linear(2, sin_embedding_size)
        with torch.no_grad():
            self.B.weight[:(sin_embedding_size)//2,0] = 28*torch.arange(0,(sin_embedding_size)//2)/sin_embedding_size
            self.B.weight[(sin_embedding_size)//2:,1] = 28*torch.arange(0,(sin_embedding_size)-(sin_embedding_size)//2)/sin_embedding_size

        self.w.append(self.B)

    def forward(self, x):
        ''' takes image x of shape [batch, d, h, w] and maps to [batch, h*w, d+k] '''
        batch_size, d, h, w = x.shape
        hh = torch.arange(0,h).T
        hh = torch.transpose(torch.tile(hh,(1,1,w,1)),3,2)
        ww = torch.arange(0,w)
        ww = torch.tile(ww,(1,1,h,1))
        coords = torch.cat([hh,ww],dim=1)           # [1, 2, h, w]
        coords = torch.reshape(coords, (1,2,h*w))   # [1, 2, h*w]
        coords = torch.transpose(coords, 1,2)       # [1, h*w ,2]

        x = torch.reshape(x, (batch_size, d, h*w))  # [batch_size, d, h*w]
        x = torch.transpose(x, 1, 2)                # [batch_size, h*w, d]

        # Normalize coordinates to range [-1, 1]
        denominator = torch.tensor([h-1, w-1])
        coords = 2*coords/denominator - 1

        if GPU:
            coords = coords.cuda()

        # [1,h*w,2] -> [1,h*w,k]
        coord_embedding = torch.sin(self.B(coords))

        coord_embedding = torch.tile(coord_embedding, (batch_size,1,1)) # [batch_size, h*w, k]
        coords = torch.tile(coords, (batch_size,1,1))

        x_with_embeddings = torch.cat([x,3*x+coord_embedding, coords],dim=2)    # [batch_size, h*w, d+k]
        #x_weight = 0.9
        #x_with_embeddings = 3*x + coord_embedding    # [batch_size, h*w, k]
        return x_with_embeddings


class Coordinet(nn.Module):
    def __init__(self, input_dhw, embedding_size, latent_size, output_size, 
                encode_hidden_sizes,decode_hidden_sizes):
        ''' 
        arguments: 
            3tuple<int>, int, int, list<int>, list<int>
        '''
        super().__init__()

        self.embed_pixels = PixelCoordinateEmbeddings(embedding_size) 

        d = input_dhw[0]
        self.embedding_to_latent_net = FullyConnected(embedding_size, latent_size, encode_hidden_sizes)

        self.latent_to_output_net = FullyConnected(latent_size, output_size, decode_hidden_sizes)

        self.w = nn.ModuleList() # list of modules
        # add to moduleList
        self.w.append(self.embed_pixels)
        self.w.append(self.embedding_to_latent_net)
        self.w.append(self.latent_to_output_net)

    def forward(self, x):
        x = self.embed_pixels(x) # [batch_size, h*w, d+k]

        # (pixel -> latent) module
        # [batch, h*w, d+k] -> [batch, h*w, latent_size]
        batch, hxw, depth = x.shape
        x = torch.reshape(x, (-1, depth))
        latent = self.embedding_to_latent_net(x)
        latent = torch.reshape(latent,(batch,hxw,-1))
        latent = torch.mean(latent, dim=1)

        # (latent -> output) module
        # [batch, latent_size] -> [batch, output]
        output = self.latent_to_output_net(latent)

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
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

        scheduler.step(total_loss/total_images)

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

BATCH_SIZE = 64
NUM_EPOCHS = 200
LR = 0.0100
MOMENTUM = 0.9
LOSS_THRESHOLD = 0.0
BREAK_TRAINING = False
GPU = torch.cuda.is_available()

##############################
# Run test
input_dhw = (1,28,28)

#model = Coordinet(input_dhw, 20, 30, 10, [],[20])
model = Coordinet(input_dhw, 50, 200, 10, [200,200],[200,200])
print(model.num_parameters())

if GPU:
    model = model.cuda()
stats1 = training_loop(model,train1,test1)


############################
# Save results (can be turned into graphs later)

hyperparams=dict(
                 BATCH_SIZE=BATCH_SIZE, 
                 NUM_EPOCHS=NUM_EPOCHS, 
                 LR=LR, 
                 MOMENTUM=MOMENTUM,
                )
stats1.update(hyperparams)

timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
experiment_name = "initial_test"
filename = f"{experiment_name}-{timestamp}.json"
json.dump(stats1, open(filename, 'w'))

try:
    from google.colab import files
    files.download(filename)
except:
    pass
