import os.path
import json
from pathlib import Path
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path: str, label_path: str, batch_size: int, image_size: list[int], rotation: bool = False, mirroring: bool = False, shuffle: bool = False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.image_files = sorted(os.listdir(file_path)) #TODO: order is not guaranteed and thus "non-determnistic"
        # load in the images from the "file_path" folder
        self.images_list = []
        for file in self.image_files:
            img = np.load(os.path.join(file_path, file))
            self.images_list.append(img)
        self.images : np.ndarray = np.array(self.images_list)
        
        self.labels = {}
        with open(f"{label_path}", "r") as f:
            self.labels = json.load(f) #key is filename, value is label
        
        self.num_batches = len(self.image_files) // self.batch_size
        self.epoch = 0
        self.current_index = 0 #dataset level index that keeps track at which index of the dataset we currently are
        

    def next(self) -> tuple | None:
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        
        """returns one batch of the provided dataset as a tuple
        (images, labels), where images represents a batch of images and labels an array with the
        corresponding labels, when called. Each image of your data set should be included only
        once in those batches until the end of one epoch. One epoch describes a run through the
        whole data set.
        
        #TODO: handle resizing , use skimage.transform.resize, this is not a flag, but rather we need to check if the images are in the same size as specified in the constructor. 
        • Note: Sometimes the images fed into a neural network are first resized. Therefore, a
        resizing option should be included within the next() method. Do not confuse resizing
        with reshaping! Resizing usually involves interpolation of data, reshaping is the simple
        reordering of data."""
        
        #TODO: handle epochs, i.e. detect when one epoch is finished (all batches have been returned) and increment epoch counter
        
        print(self.current_index)
        
        if self.current_index >= len(self.images): 
            print("Epoch is complete. Starting with next epoch! \n")
            #TODO: reshuflling would need to be applied here
            self.current_index = 0 #start from the beginning of the dataset
            self.epoch += 1

        #TODO: Make sure all your batches have the same size. If the last batch is smaller than the others, complete that batch by reusing images from the beginning of your training data set.
        #TODO: handle last batch in the case that last batch is smaller than batch_size (loop back to first batch and reuse as many images necessary to fill it up)

        images = []
        labels = []
        
        remaining = len(self.images) - self.current_index 
        # TODO: we need to check if there are enough samples left in the data to fill up the last batch 
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
        
        #TODO: get the corresponding labels(just the int) out of the file names (just cut of the ".npy" part), that string that is the key for the labels dict
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

