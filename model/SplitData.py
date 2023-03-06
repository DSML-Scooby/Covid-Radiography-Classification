import os
import splitfolders

folder_path = '../data/COVID-19_Radiography_Dataset/TwoClasses/'

data = os.listdir(folder_path)

# Function to split folders in the input to output with specified ratio of train, test, val
splitfolders.ratio(folder_path, # The location of dataset
                   output="../data/COVID-19_Radiography_Dataset/TwoClasses/split", # The output location
                   seed=42, # The number of seed
                   ratio=(.6, .2, .2), # The ratio of split dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
                   )