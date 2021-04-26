#!/usr/bin/python3

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pickle
import interrupt
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
                                               batch_size=batch_size, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False)

    # set up for training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,)# momentum=momentum)
    loss_criterion = nn.CrossEntropyLoss()

    #  print("Num Parameters: ", model.num_parameters())

    curr_loss = 10e8

    # training loop
    epoch = 0
    while(curr_loss > LOSS_THRESHOLD):
        curr_loss = 0
        num_training_images = 0
        for i, (X, y) in enumerate(train_loader):
            
            #  Forward pass
            outputs = model(X)
            loss = loss_criterion(outputs, y)
            curr_loss += loss.item()*y.shape[0] #curr batch size
            num_training_images += y.shape[0]
            
            # Backprpagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        curr_loss /= num_training_images
        print(f"\r Epoch {epoch+1}, Average Loss: {curr_loss}",end='')
        epoch += 1
        if(BREAK_TRAINING): 
            break
        if(epoch > 10000):
            print("*********************** ERROR ******************************")
            print(" ************************ got stuck ***********************")
            break
    print()

    # testing
    with torch.no_grad():
        total_correct = 0
        total_loss = 0
        total_images = 0
        for X, y in train_loader:
            outputs = model(X)

            prediction = torch.argmax(outputs, dim=1) # max outputs max and argmax
            num_correct = torch.sum(prediction == y)
            total_correct += num_correct.item()
            total_images += y.shape[0]

            total_loss += loss_criterion(outputs, y).item()*y.shape[0]

        train_accuracy = total_correct / total_images
        train_loss = total_loss / total_images
        num_training_images = total_images

        total_correct = 0
        total_loss = 0
        total_images = 0
        for X, y in test_loader:
            outputs = model(X)

            prediction = torch.argmax(outputs, dim=1) # max outputs max and argmax
            num_correct = torch.sum(prediction == y)
            total_correct += num_correct.item()
            total_images += y.shape[0]

            total_loss += loss_criterion(outputs, y).item()*y.shape[0]

        gen_accuracy = total_correct / total_images
        gen_loss = total_loss / total_images
        num_test_images = total_images
        
        print("Train Accuracy: ", train_accuracy)
        print("Test Accuracy: ", gen_accuracy)
        #  print("Test Loss: ", gen_loss)
        #  print("\n\n")

    stats = dict(num_parameters = model.num_parameters(), 
                 train_accuracy = train_accuracy,
                 train_loss = train_loss,
                 generalisation_accuracy = gen_accuracy,
                 generalisation_loss = gen_loss,
                 num_training_images = num_training_images,
                 num_test_images = num_test_images)
    return stats

def flatten(x):
    return torch.reshape(x,(-1,))

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #  flatten,
    ])

train1 = torchvision.datasets.FashionMNIST('~/data/',train=True,transform=transform, download=True)
test1 = torchvision.datasets.FashionMNIST('~/data/',train=False,transform=transform, download=True)

# hyperparams

batch_size = 256
num_epochs = 50
lr = 0.0001
momentum = 0.9
LOSS_THRESHOLD = 0.98
BREAK_TRAINING = False
DATASET_SIZE = len(train1)


# get data
stats_list = []
accuracy_list = []
# iterate over experiment variations 
for i in [0]:
    exp1_acc = []
    # run 10 of each experiment
    for _ in range(10):
        model = Coordinet(3, 30, 10, [],[20])
        stats1 = training_loop(model,train1,test1)
        exp1_acc.append(stats1['generalisation_accuracy'])
        stats_list.append([DATASET_SIZE,stats1])

    accuracies_dict = {'n':DATASET_SIZE, 1:exp1_acc, }
    print(accuracies_dict)
    print([np.mean(accuracies_dict[i]) for i in [1,]])
    accuracy_list.append(accuracies_dict)


    with open("accuracy_list.pickle", "wb") as f:
        pickle.dump(accuracy_list,f)

    with open("stats.pickle", "wb") as f:
        pickle.dump(stats_list, f)
with open("accuracy_list.pickle", "wb") as f:
    pickle.dump(accuracy_list,f)

with open("stats.pickle", "wb") as f:
    pickle.dump(stats_list, f)
