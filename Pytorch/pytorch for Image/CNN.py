import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# import conx as cx
# mnist = cx.Dataset.get('mnist')
# mnist.info()
# digits = mnist.inputs.select(lambda i,ds: ds.labels[i] , slice=20)
# cx.view(digits)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # Conv2D(inputFilters/channels, output_channels, kernel_Size)
    self.conv1 = nn.Conv2d(1, 6, 3)
    self.conv2 = nn.Conv2d(6, 16, 3)
    # an affine operation: y = Wx + b
    self.fc1 = nn.Linear(16*6*6, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    # max pooling window 2 * 2
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # I can write only to instead of 2*2 as it's a square size
    x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x

  def num_flat_features(self, x):
    size = x.size()[1:] # all dimention except batch dimention
    num_feature = 1
    for s in size:
      num_feature *= s
    
    return num_feature

net = Net()
print(net)

params = list(net.parameters()) # .parameters returns learnable parameters
print("Learnable steps: ", len(params))
for p in params:
  print("Learnable parameter per step:", p.size())

## feed a image in dataset 
img_path = "/content/6.png"
image = Image.open(img_path)
trans = transforms.ToPILImage()
trans1 = transforms.ToTensor()
plt.imshow(trans(trans1(image)))

x = trans1(image)
print("Image Size: ", x.size())
input = x.view(-1, 1, 32, 32)
print("Input dims: ", input.size())
out = net(input)

print("\n\nOutput\n")
print(out)
net.zero_grad()
out.backward(torch.randn(1, 10))
print(out)

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()
loss = criterion(output, target)
print("Loss: ", loss)
print("Loss Gradient Tracking: \n")
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])


## Back Propagation
 

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# Updating the Weights
# SGD (Stochastic Gradient Descent): weight = weight - learning rate * gradient

## If we want to do it menually 
# learning_rate = 0.1
# for f in net.parameters():
#   f.data.sub_(f.grad.data * learning_rate)

# But pytorch has function for it

import torch.optim as optim

# Creating Optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.03)

# in training loop
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
print("Output before optimization: ", output)


optimizer.step()

output = net(input)
loss = criterion(output, target)
loss.backward()
print("Output after optimization: ", output)

