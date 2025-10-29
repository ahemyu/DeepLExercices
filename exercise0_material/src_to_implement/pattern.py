import numpy as np
import matplotlib.pyplot as plt

class Checker: 
    """The first pattern to implement is a checkerboard pattern in the class Checker with adaptable
    tile size and resolution. You might want to start with a fixed tile size and adapt later on. For
    simplicity we assume that the resolution is divisible by the tile size without remainder."""
    
    def __init__(self, resolution: int, tile_size: int):
        
        self.resolution = resolution # defines number of pixels in each dimension
        self.tile_size = tile_size # defines the number of pixel an individual tile has in each dimension.
        self.output: np.ndarray = np.ndarray(shape=(self.resolution, self.resolution))

    def draw(self):
        """creates the checkerboard pattern as a numpy array.
        The tile in the top left corner should be black. In order to avoid truncated checkerboard
        patterns, make sure your code only allows values for resolution that are evenly dividable
        by 2 * tile size."""

        # if resolution is not evenly divisible by 2, return
        if not (self.resolution % (2 * self.tile_size) == 0):
            return None

        # Represent black as 1 and white as 0
        # determine the shape of output based on resolution and tile_size 
        # ok so output shape will be (resolution, resolution)
        # tile_size determines how big of a square each tile is, i.e. for tile_size of 2 each tile is 2x2 
        # no loops allowed so only np functions 
        
        # np.tile(), np.arange(), np.zeros(), np.ones(), np.concatenate() and np.expand dims()
        
        # as we start with black always, we can always draw the first line already: 
        
        #use np.arange to create the exactly the amount of ones we need in the first line (wqhich is exactly resokutiuon / tile_size / 2)

        # we first need a mapping of row and column to tile_row, tile_col 
        # then we can use those indices to determine the color of that file, it is black if sum of indices even and white if uneven
        
        #TODO: based on array cordinate determine the tile_row and col 
        # tile_row_index = row_index // tile_size
        # tile_col_index = col_index // tile_size
        
        # (tile_row_index + tile_col_index) % 2 != 0 => white <=> 0 
        # (tile_row_index + tile_col_index) % 2 == 0 => black <=> 1 
        
        # use np.indices to get column_index anmd row_index arrays
        
        indices = np.indices(self.output.shape)
        
        row_indices = indices[0]
        col_indices = indices[1]

        tile_row_indices = row_indices // self.tile_size
        tile_col_indices = col_indices // self.tile_size
        
        index_sums = tile_col_indices + tile_row_indices
        
        self.output = index_sums % 2# now black is 0 and white is 1 
        
        print("OUTPUT: \n", self.output)
        return self.output.copy() #numpy arrays are always passed per reference
        
        
        

    def show(self):
        """shows the checkerboard pattern with for example
        plt.imshow(). If you want to display a grayscale image you can use cmap = gray as
        a parameter for this function."""

        plt.imshow(self.output, cmap="gray")
        

class Circle: 
    def __init__(self):
        pass
    
    def draw(self):
        pass

    def show(self):
        pass