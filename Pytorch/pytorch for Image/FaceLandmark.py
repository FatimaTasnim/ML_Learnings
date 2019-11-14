from __future__ import print_function, division 
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
import zipfile
warnings.filterwarnings("ignore")

plt.ion() # Interactive mode
zip_ref = zipfile.ZipFile("/content/faces.zip", 'r')
zip_ref.extractall("/content/allfaces")
zip_ref.close()
landmarks_frame = pd.read_csv('/content/allfaces/faces/face_landmarks.csv')
print(landmarks_frame.shape)

n = 56
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)
path = "/content/allfaces/faces/" + img_name
print('image name: {}' .format(img_name))
print("landmarks shape: {}" .format(landmarks.shape))
print("First 4 landmarks: {}".format(landmarks[:4]))

def show_landmarks(image, landmarks):
  plt.imshow(image)
  plt.scatter(landmarks[:, 0], landmarks[:, 1], s = 5, marker = '*', c = 'b')
  plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()
show_landmarks(io.imread(path), landmarks)
plt.show()
temp = 1

class FaceLandmarksDataset(Dataset):
  def __init__(self, csv_file, root_dir, transform=None):
    print("hi there")
    self.landmarks_frame = pd.read_csv(csv_file) # string path 
    self.root_dir = root_dir
    self.transform = transform 

  def __len__(self):
    return len(self.landmarks_frame)

  def __getitem__(self, idx):
    print("Index: ", idx)
    if torch.is_tensor(idx):
      idx = idx.tolist()
  
    img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])

    image = io.imread(img_name)
    landmarks = self.landmarks_frame.iloc[idx, 1:]
    landmarks = np.array([landmarks])
    landmarks = landmarks.astype('float').reshape(-1, 2)
    sample = {'image': image, 'landmarks': landmarks}

    if self.transform:
      sample = self.transform(sample)

    return sample


face_dataset = FaceLandmarksDataset(csv_file = '/content/allfaces/faces/face_landmarks.csv', root_dir = '/content/allfaces/faces/')
fig = plt.figure
print(face_dataset)
for i in range(len(face_dataset)):
  sample = face_dataset[i]
  print(i, sample['image'].shape, sample['landmarks'].shape)
       
  ax = plt.subplot(1, 4, i+1)
  plt.tight_layout()
  ax.set_title('Sample #{}'.format(i))
  ax.axis('off')
  show_landmarks(**sample)
  if i==3 :
    plt.show()
    break

# tsfm = transform(params)
# transformed_sample = tsfm(sample)


class Rescale(object):
  # Rescale the image in a sample to a given size

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, sample):
    image, landmarks = sample['image'], sample['landmarks']

    h, w = image.shape[:2]
    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size , self.output_size * w / h
    else:
      new_h, new_w = self.output_size
    
    new_h, new_w = int(new_h), int(new_w)

    img = transform.resize(image, (new_h, new_w))

    # h and w are swaped for landmarks because for images
    # x and u axes are axis 1 and 0 respectively

    landmarks = landmarks * [new_w/w, new_h/h]
    return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
  """
  crop randomly the iumage in a sample
  args: output_soze(tuple or int): Desired output sioze. if int, square crop is made

  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size)
    else:
      assert len(output_size) == 2
      self.output_size = output_size

  def __call__(self, sample):
    image, lanmarrks = sample['image'], sample['landmarks']
    h, w  = image.shape[:2]
    new_h, new_w = self.output_size
    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    image = image[top: top + new_h, left: left + new_w]

    landmak = landmarks - [left - top]
    
    return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
  def __call__(self, sample):
    image, landmarks = sample['image'], sample['landmarks']

    # swap color axis cause numpy image HWC and torch image CHW
    image = image.transpose((2,0,1))
    return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}


### Compose Transforms
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256), RandomCrop(224)])

# Apply each of the above transform on the sample image

fig = plt.figure()
sample = face_dataset[56]
for i, tsfm in enumerate([scale, crop, composed]):
  transformed_sample = tsfm(sample)
  ax = plt.subplot(1, 3, i+1)
  plt.tight_layout()
  ax.set_title(type(tsfm).__name__)
  show_landmarks(**transformed_sample)

plt.show()

### Itereting through the dataset
transformed_dataset = FaceLandmarksDataset(csv_file = '/content/allfaces/faces/face_landmarks.csv',
                                           root_dir = '/content/allfaces/faces/',
                                           transform = transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()                   
                                           ]))
for i in range(len(transformed_dataset)):
  sample = transformed_dataset[i]
  print(i, sample['image'].size(), sample['landmarks'].size())

  if i==3:
    break


"""
important things to do 
- batching the data
- shuffling the data
- load the data in parallel using "Multiprocessing " Workers

torch.utils.data.DataLoader provides these feature
"""

dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True)

def show_landmarks_batch(sample_batched):
  image_batch, landmarks_batch =  sample_batched['image'], sample_batched['landmarks']
  batch_size = len(image_batch)
  im_size = image_batch.size(2)
  grid_bordered_size = 2

  grid = utils.make_grid(image_batch)
  plt.imshow(grid.numpy().transpose((1, 2, 0)))

  for i in range(batch_size):
    plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i+1) * grid_bordered_size,
                landmarks_batch[i, :, 1].numpy() + grid_bordered_size,
                s = 10, marker = '.', c = 'r')
    plt.title("batch from dataloader")

for i_batch, sample_batched in enumerate(dataloader):
  print(i_batch, sample_batched['image'].size(),
        sample_batched['landmarks'].size())
  
        # observe 4th batch and stop
  if i_batch == 3:
    plt.figure()
    show_landmarks_batch(sample_batched)
    plt.axis('off')
    plt.ioff()
    plt.show()
    break