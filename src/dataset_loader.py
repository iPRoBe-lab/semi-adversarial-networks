############################################
#          Semi-Adversarial Network        #
#               (data_loader)              #
#               iPRoBe lab                 #
#                                          #
############################################

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import pandas as pd
import pyprind
import sys
from PIL import Image
import os


class CelebaDataset(Dataset):
    def __init__(self, image_path, proto_smG_path, proto_opG_path,
                 metadata_path, transform, mode):
        self.image_path = image_path
        self.proto_smG_path = proto_smG_path
        self.proto_opG_path = proto_opG_path
        self.transform = transform
        self.proto_transform = transforms.ToTensor()
        self.mode = mode
        df = pd.read_csv(metadata_path, sep='\s+', skiprows=1)
        df.Male = df.Male.map({-1: 0, 1: 1})
        self.df = df.reset_index()
        #self.flip_rate = flip_rate

        print('Start preprocessing dataset..!')
        self.preprocess()
        print('Finished preprocessing dataset..!')

        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)

        print('******', self.num_data)

    def preprocess(self):
        image_files = set(os.listdir(self.image_path))
        protoSM_files = set(os.listdir(self.proto_smG_path))
        protoOP_files = set(os.listdir(self.proto_opG_path))
        existing_files = image_files & protoSM_files & protoOP_files
        print('existing_files : ', len(existing_files))

        self.train_filenames = []
        self.train_labels = []
        self.test_filenames = []
        self.test_labels = []

        #df = self.df.sample(frac=1).reset_index(drop=True)
        pbar = pyprind.ProgBar(len(self.df))
        for row in self.df.iterrows():
            filename = row[1]['index']
            gender = row[1]['Male']

            if self.mode == 'train':
                if filename in existing_files:
                    self.train_filenames.append(filename)
                    self.train_labels.append(gender)
            elif self.mode == 'test':
                if filename in existing_files:
                    self.test_filenames.append(filename)
                    self.test_labels.append(gender)
            pbar.update()
        sys.stderr.flush()

    def __getitem__(self, index):
        if self.mode == 'train':
            fname = self.train_filenames[index]
            image = Image.open(os.path.join(self.image_path, fname))
            smG_proto = Image.open(os.path.join(self.proto_smG_path, fname))
            opG_proto = Image.open(os.path.join(self.proto_opG_path, fname))

            label = (self.train_labels[index],)

            return (self.transform(image), self.proto_transform(smG_proto),
                    self.proto_transform(opG_proto), torch.LongTensor(label))

        elif self.mode == 'test':
            image = Image.open(os.path.join(self.image_path,
                                            self.test_filenames[index]))
            label = (self.test_labels[index],)

            return self.transform(image), torch.LongTensor(label)

    def __len__(self):
        return self.num_data


def get_loader(image_path, proto_same_path, proto_oppo_path, metadata_path,
               crop_size=(224, 224), image_size=(224, 224), batch_size=64,
               dataset='CelebA', mode='train',
               num_workers=1):
    """Build and return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomCrop(size=crop_size),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    #if dataset == 'CelebA':
    dataset = CelebaDataset(image_path, proto_same_path, proto_oppo_path,
                            metadata_path, transform, mode) #, flip_rate=flip_rate)

    if mode == 'train':
        shuffle = True
    else:
        shuffle = False

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)
    return data_loader
