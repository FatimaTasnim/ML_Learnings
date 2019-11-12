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

    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16*25, 120) # 16 * 5 * 5
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
     x = self.pool(F.relu(self.conv1(x)))
     x = self.pool(F.relu(self.conv2(x)))
     x = x.view(-1, 16*25)
     x = F.relu(self.fc1(x))
     x = F.relu(self.fc2(x))
     x = self.fc3(x)
     return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net.to(device) # this will recursively go over all modules asn convert their parameters and buffers to CUDA tensors

PATH = './cifar_net.pth'

net.load_state_dict(torch.load(PATH))
dummuydata = iter(testloader)
data = dummydata.next()
images, labels = data[0].to(device), data[1].to(device) # need to send the inputs and targets to gpu at each step


outputs = net(images)
print(outputs.data)
_, predicted = torch.max(outputs.data, 1)


imshow(torchvision.utils.make_grid(images.to("cpu"))) # imshow can not work on gpu tensors so using cpu tensors 

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

