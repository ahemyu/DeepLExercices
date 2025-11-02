import os.path
import json
from pathlib import Path
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import skimage

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path: str, label_path: str, batch_size: int, image_size: list[int], rotation: bool = False, mirroring: bool = False, shuffle: bool = False):
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.image_files = sorted(os.listdir(file_path)) 
        # load in the images from the "file_path" folder
        self.images_list = []
        target_height, target_width, target_channels = self.image_size
        for file in self.image_files:
            img = np.load(os.path.join(file_path, file))
            # resize if needed
            if (img.shape[0] != target_height or 
                img.shape[1] != target_width):
                img = skimage.transform.resize(img, 
                            (target_height, target_width, target_channels),
                            anti_aliasing=True, 
                            preserve_range=True)
            self.images_list.append(img)

        self.images : np.ndarray = np.array(self.images_list)
        
        self.labels = {}
        with open(f"{label_path}", "r") as f:
            self.labels = json.load(f) #key is filename, value is label
        
        self.epoch = 0
        self.current_index = 0 #dataset level index that keeps track at which index of the dataset we currently are
        self.indices = np.arange(len(self.images)) # these are the indices we are going to use to implement shuffling
        
        if self.shuffle:
            np.random.shuffle(self.indices)
        

    def next(self) -> tuple:
        if self.current_index >= len(self.images): 
            self.current_index = 0 #start from the beginning of the dataset
            self.epoch += 1
            if self.shuffle:
                np.random.shuffle(self.indices)

        images = []
        labels = []
        remaining = len(self.images) - self.current_index 
        
        # we need to check if there are enough samples left in the data to fill up the last batch 
        if remaining < self.batch_size:
            # we are in the last batch that has too few elements
            # take al of the images we have
            remaining_indices = self.indices[self.current_index: ] # all that we have
            # add missing ones from the start
            missing_count = self.batch_size - remaining
            missing_indices = self.indices[0 : missing_count]
            batch_indices = np.concatenate([remaining_indices, missing_indices], axis=0)
            images = self.images[batch_indices]
            # handle labels in the same way
            batch_filenames = [self.image_files[idx] for idx in batch_indices]
            batch_keys = [Path(f).stem for f in batch_filenames]
            labels = np.array([self.labels[k] for k in batch_keys], dtype=int)
            
            # add last batch_size to self.current_index such that in next call it will be reset and epoch increased correctly
            self.current_index += self.batch_size
            augmented = np.array([self.augment(img) for img in images])
            
            return augmented, labels
            
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        
        # use those indices to get the actual images
        images = self.images[batch_indices]
        
        # use the same indices to get filenames and then labels
        batch_filenames = [self.image_files[idx] for idx in batch_indices]
        batch_keys = [Path(f).stem for f in batch_filenames]
        labels = np.array([self.labels[k] for k in batch_keys], dtype=int)
        
        self.current_index += self.batch_size

        
        augmented = np.array([self.augment(img) for img in images])
        return augmented, labels


    def augment(self, img):
        augmented_img = img.copy()
        
        if self.mirroring:
            # randomly decide to mirror or not (50% chance)
            if np.random.rand() > 0.5:
                # randomly choode between types of rotation
                mirror_type = np.random.choice(['horizontal', 'vertical', 'both'])
                
                if mirror_type == 'horizontal':
                    augmented_img = np.flip(augmented_img, axis=1)
                elif mirror_type == 'vertical':
                    augmented_img = np.flip(augmented_img, axis=0)
                else:  # both
                    augmented_img = np.flip(augmented_img, axis=1)
                    augmented_img = np.flip(augmented_img, axis=0)
        
        if self.rotation:
            # also include not rotating
            angle = np.random.choice([0, 90, 180, 270])
            
            if angle != 0:
                # np.rot90 rotates by 90 degrees k times
                k = angle // 90
                augmented_img = np.rot90(augmented_img, k=k)
        
        return augmented_img


    def current_epoch(self):
        # return the current epoch number
        return self.epoch


    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]

    def show(self):
        # get a batch of images and lables
        images, labels = self.next()

        batch_size = images.shape[0]
        # roughly square outline
        cols = int(np.ceil(np.sqrt(batch_size)))
        rows = int(np.ceil(batch_size / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()
        
        for idx in range(batch_size):
            # normalize between 0 and 1
            axes[idx].imshow(images[idx] / 255.0)
            axes[idx].set_title(self.class_name(labels[idx]))
            axes[idx].axis('off')
        
        plt.show()