
import pandas as pd 
import numpy as np 


from sklearn.model_selection import train_test_split 

import torch 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms 

from config import * 


class SignDataset(Dataset):

  def __init__(self, images, labels, transform = None):
    self.images = images 
    self.labels = labels 
    self.transform = transform 

  
  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image = self.images[idx].astype(np.uint8)
    label = self.labels[idx]

    if self.transform:
      image = self.transform(image)

    return image, label


def get_loaders():

  train_df = pd.read_csv(TRAIN_CSV)
  test_df = pd.read_csv(TEST_CSV)

  X_train = train_df.drop('label', axis = 1).values 

  X_test = test_df.drop('label', axis = 1).values 
  
  y_train = train_df["label"].values
  y_test = test_df["label"].values

  # Remap labels to 0–23

  label_map = {
      0:0,   # A
      1:1,   # B
      2:2,
      3:3,
      4:4,
      5:5,
      6:6,
      7:7,
      8:8,   # I
      10:9,  # K
      11:10,
      12:11,
      13:12,
      14:13,
      15:14,
      16:15,
      17:16,
      18:17,
      19:18,
      20:19,
      21:20,
      22:21,
      23:22,
      24:23  # Y
  }

  y_train = np.array(
      [label_map[label] for label in y_train]
  )

  y_test = np.array(
      [label_map[label] for label in y_test]
  )

  #reshaping 

  X_train = X_train.reshape(-1,28,28)
  X_test = X_test.reshape(-1,28,28)

  #creating validation data 

  X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size = 0.2, random_state = 42, stratify = y_train
  )

  #augmentations 

  train_transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    
    transforms.Grayscale(num_output_channels = 3),

    transforms.RandomRotation(10),

    transforms.RandomAffine(
      degrees = 0,
      translate = (0.1,0.1),
      scale = (0.9, 1.1)
    ),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    
    transforms.RandomHorizontalFlip(),

    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),

    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
    )
  ])
  
  val_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    
    transforms.Grayscale(num_output_channels = 3),

    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
    )
  ])

  #creating datasets 

  train_dataset = SignDataset(X_train,y_train, train_transform)
  val_dataset = SignDataset(X_val, y_val, val_transform)
  test_dataset = SignDataset(X_test, y_test, val_transform)

  #creating dataloaders

  train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
  val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
  test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)

  return train_loader, val_loader, test_loader