import numpy as np
import matplotlib.pyplot as plt

#rectangular differences - two rectangular areas - histograms of oriented gradients - compare to our method  - HAAR Features

# 1. Load the text file into a numpy array
# Note: If your numbers are separated by spaces instead of commas, 
# change delimiter=',' to delimiter=' '
print("Loading pixel data from combined48.txt...")
raw_data = np.loadtxt('famous48-face-recognition-AI-project\data\combined48.txt')


# 2. Reshape the flat data into 2D images
# raw_data.shape[0] gives you the total number of images (N)
num_images = raw_data.shape[0]
print(f"Found {num_images} images. Reshaping to 24x24...")

# First, let's confirm the shape. It should print (6835, 584)
print(f"Raw data shape is: {raw_data.shape}")

# SLICE THE DATA
# If your 8 labels are at the END of the row (most common):
pixel_data = raw_data[:, :576]

# Note: If the image looks like pure static/noise when it plots, 
# that means the labels were actually at the BEGINNING. 
# If so, comment out the line above and use this one instead:
# pixel_data = raw_data[:, 8:]

# Now reshape just the 576 pixel columns!
images = pixel_data.reshape((num_images, 24, 24))

# 3. Compute the average face across all images
average_face = np.mean(images, axis=0)

# 4. Plot the result with the coordinate grid
plt.figure(figsize=(8, 8))
plt.imshow(average_face, cmap='gray')

# Add grid lines to map out your row_start:row_end, col_start:col_end
plt.grid(True, which='both', color='red', linewidth=0.5)
plt.xticks(np.arange(0, 24, 1))
plt.yticks(np.arange(0, 24, 1))
plt.title(f"Average Face (Computed from {num_images} images)")

# Show the plot
plt.show()