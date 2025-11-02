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
        

    def next(self) -> tuple | None:
        #TODO: shuffling: If the shuffle flag is True, the order of your data set (= order in which the images appear) is random (Not only the order inside one batch!).
        # Note: With shuffling, the ImageGenerator must not return duplicates within one epoch. → If your index reaches the end of your data during batch creation reset your
        # index to point towards the first elements of your dataset and shuffle your indices again after one epoch.
        if self.current_index >= len(self.images): 
            print("Epoch is complete. Starting with next epoch! \n")
            #TODO: reshuflling would need to be applied here
            self.current_index = 0 #start from the beginning of the dataset
            self.epoch += 1

        images = []
        labels = []
        remaining = len(self.images) - self.current_index 
        
        # we need to check if there are enough samples left in the data to fill up the last batch 
        if remaining < self.batch_size:
            # take al of the images we have
            remaining_imgs = self.images[self.current_index: ] # all that we have
            # add missing ones from the start
            missing_count = self.batch_size - remaining
            missing_imgs = self.images[0 : missing_count]
            images = np.concatenate([remaining_imgs, missing_imgs], axis=0)
            # handle labels in the same way
            image_file_names_with_endings = self.image_files[self.current_index:]
            remaining_keys = [Path(f).stem for f in image_file_names_with_endings]

            # how many more we need from the start
            missing_count = self.batch_size - remaining
            missing_file_names = self.image_files[:missing_count]
            missing_keys = [Path(f).stem for f in missing_file_names]

            # concatenate remaining + missing keys into one flat batch
            all_keys = remaining_keys + missing_keys
            labels = np.array([self.labels[k] for k in all_keys], dtype=int)
            
            # add last batch_size to self.current_index such that in next call it will be reset and epoch increased correctly
            self.current_index += self.batch_size
            
            return images, labels
            
        images = self.images[self.current_index: self.current_index + self.batch_size]
        
        # get the corresponding labels(just the int) out of the file names (just cut of the ".npy" part), that string that is the key for the labels dict
        image_file_names_with_endings = self.image_files[self.current_index : self.current_index +  self.batch_size]
        image_file_keys = [Path(f).stem for f in image_file_names_with_endings]
        labels = [self.labels[key] for key in image_file_keys]

        self.current_index += self.batch_size

        return images, labels


    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        return img


    def current_epoch(self):
        # return the current epoch number
        return self.epoch


    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return


    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method 
        pass

