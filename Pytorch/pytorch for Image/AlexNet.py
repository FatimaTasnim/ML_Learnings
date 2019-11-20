import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

transforms = transforms.Compose(
    [transforms.ToTensor(),
     # transform.normalize (mean for all channels) (std for all channels)
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # (image-mean)/std
)

trainset = torchvision.datasets.CIFAR10(root = './data', train=True, download=True, transform=transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root = './data', train=False, download = True, transform = transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False, num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
  img = img / 2 + 0.5 ## unnormalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

dummydata = iter(trainloader)

images, labels = dummydata.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' %classes[labels[j]] for j in range (4)))


## convnet 
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # Input_ch = 3, output_ch = 64, Kernel_size = 5, stride = 4, padding = 3
    self.conv1 = nn.Conv2d(3, 64, 5, stride = 4, padding = 2)
    self.conv2 = nn.Conv2d(64, 192, 5, padding = 2)
    self.conv3 = nn.Conv2d(192, 384, 3, padding = 1)
    self.conv4 = nn.Conv2d(384, 256, 3, padding = 1)
    self.conv5 = nn.Conv2d(256, 256, 3, padding = 2)
    
    self.fc1 = nn.Linear(256 * 6 * 6, 1800) # 16 * 5 * 5
    self.fc2 = nn.Linear(1800, 512)
    self.fc3 = nn.Linear(512, 10)
    self.pool = nn.MaxPool2d(kernel_size= 3, stride=2)
    self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

  def forward(self, x):
     x = self.pool(F.relu(self.conv1(x), inplace = True))
     #print("Conv1: ", x.shape)
     x = self.pool(F.relu(self.conv2(x), inplace = True))
     #print("Conv2: ", x.shape)
     x = F.relu(self.conv3(x), inplace = True)
     #print("Conv3: ", x.shape)
     x = F.relu(self.conv4(x), inplace = True)
     #print("Conv4: ", x.shape)
     x = self.pool(F.relu(self.conv5(x), inplace = True))
     #print("Conv5: ", x.shape)
     x = self.avgpool(x)
     x = x.view(-1, 256 * 6 * 6)
     x = F.relu(self.fc1(x))
     x = F.relu(self.fc2(x))
     x = self.fc3(x)
     return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# Why Momentum works https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d

# Training Time

for epoch in range(2):
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % 2000 == 1999:
       if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dummuydata = iter(testloader)
images, labels = dummydata.next()

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))