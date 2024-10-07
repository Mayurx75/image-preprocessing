#!/usr/bin/env python
# coding: utf-8

# In[12]:


import cv2

# Load the image
image = cv2.imread('C:\Users\kumar\OneDrive\Desktop\lab1\wallpaperflare.com_wallpaper (10).jpg',cv2.IMREAD_GRAYSCALE)

# Find the minimum and maximum pixel values
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)

print(f'Minimum pixel value: {min_val}')
print(f'Maximum pixel value: {max_val}')


# In[15]:


pip install opencv-python


# In[17]:


import cv2

# Load the image
image = cv2.imread('C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg',cv2.IMREAD_GRAYSCALE)

# Find the minimum and maximum pixel values
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)

print(f'Minimum pixel value: {min_val}')
print(f'Maximum pixel value: {max_val}')


# In[19]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
image = cv2.imread('C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg')

# Function to display images
def show_image(title, img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Rotate the image by 90 degrees
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
show_image('Rotated Image', rotated)

# Flip the image horizontally
flipped = cv2.flip(image, 1)
show_image('Flipped Image', flipped)

# Crop the image
cropped = image[50:200, 50:200]
show_image('Cropped Image', cropped)

# Resize the image
resized = cv2.resize(image, (100, 100))
show_image('Resized Image', resized)

# Add Gaussian noise
noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
noisy_image = cv2.add(image, noise)
show_image('Noisy Image', noisy_image)

# Change brightness
brightened = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
show_image('Brightened Image', brightened)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show_image('Grayscale Image', gray)

# Shear transformation
rows, cols, ch = image.shape
pts1 = np.float32([[50,50], [200,50], [50,200]])
pts2 = np.float32([[10,100], [200,50], [100,250]])
M = cv2.getAffineTransform(pts1, pts2)
sheared = cv2.warpAffine(image, M, (cols, rows))
show_image('Sheared Image', sheared)


# In[22]:


from PIL import Image

# Open an image file
with Image.open('C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg') as img:
    # Rotate the image by 30 degrees
    rotated_img_30 = img.rotate(30, expand=True)
    rotated_img_30.save('C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg')
    
    # Rotate the image by 60 degrees
    rotated_img_60 = img.rotate(60, expand=True)
    rotated_img_60.save('C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg')
    
    # Rotate the image by 90 degrees
    rotated_img_90 = img.rotate(90, expand=True)
    rotated_img_90.save('C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg')


# In[23]:


from PIL import Image

# Load the image
image = Image.open("C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg")

# Rotate the image by 30, 60, and 90 degrees
rotated_30 = image.rotate(30, expand=True)
rotated_60 = image.rotate(60, expand=True)
rotated_90 = image.rotate(90, expand=True)

# Save the rotated images
rotated_30.save("C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg")
rotated_60.save("C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg")
rotated_90.save("C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg")

# Display the rotated images (Optional)
rotated_30.show()
rotated_60.show()
rotated_90.show()


# In[24]:


import cv2
import numpy as np

# Load the original image
image = cv2.imread('C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg')

# Function to save multiple transformed images
def generate_images(image, n):
    for i in range(n):
        transformed_image = image.copy()
        
        # Rotate the image
        angle = 10 * i  # Change the angle for each image
        M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1)
        transformed_image = cv2.warpAffine(transformed_image, M, (image.shape[1], image.shape[0]))

        # Save the transformed image
        output_path = f'output_image_{i}.jpg'
        cv2.imwrite(output_path, transformed_image)
        
        # Print the path of the saved image
        print(f'Saved: {output_path}')

# Generate 10 variations of the image
generate_images(image, 10)


# In[25]:


import matplotlib.pyplot as plt
import cv2

# Load an image using OpenCV
image = cv2.imread('C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg', 0)  # Load the image in grayscale

# Calculate histogram
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# Plot the histogram
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()


# In[26]:


import cv2
import matplotlib.pyplot as plt

# Load the original image
image = cv2.imread('C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg', cv2.IMREAD_GRAYSCALE)

# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)

# Plot histograms
plt.figure(figsize=(12, 6))

# Original Image Histogram
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Histogram of Original Image')
plt.hist(image.ravel(), bins=256, range=[0, 256], color='black')

# CLAHE Image Histogram
plt.subplot(2, 2, 3)
plt.title('CLAHE Enhanced Image')
plt.imshow(clahe_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Histogram of CLAHE Enhanced Image')
plt.hist(clahe_image.ravel(), bins=256, range=[0, 256], color='black')

plt.tight_layout()
plt.show()


# In[ ]:


import cv2
import numpy as np

# Load the image
image = cv2.imread('C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg')

# Convert to YUV color space
yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

# Equalize the histogram of the Y channel
yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])

# Convert back to BGR color space
equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

# Save or display the result
cv2.imwrite('equalized_image.jpg', equalized_image)
cv2.imshow('Contrast Enhanced Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


pip install imgaug


# In[ ]:


import imageio
import imgaug.augmenters as iaa

# Load an image
image = imageio.imread('C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg')

# Define a sequence of augmentations
seq = iaa.Sequential([
    iaa.Fliplr(0.5),     # horizontal flips
    iaa.Crop(percent=(0, 0.1)),  # random crops
    iaa.LinearContrast((0.75, 1.5)),  # improve or worsen the contrast
    iaa.Rotate((-45, 45)),  # rotate by -45 to +45 degrees
    iaa.AdditiveGaussianNoise(scale=(10, 60))  # add gaussian noise
])

# Apply the augmentations
images_aug = seq(images=[image])

# Save the augmented image
imageio.imsave('augmented_image.jpg', images_aug[0])


# In[ ]:


import cv2

# Load the image
image = cv2.imread('C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization
enhanced_image = cv2.equalizeHist(gray_image)

# Save the result
cv2.imwrite('C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg', enhanced_image)

# Display the images
cv2.imshow('Original Image', gray_image)
cv2.imshow('Contrast Enhanced Image', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import cv2
import matplotlib.pyplot as plt

# Load the original image
image = cv2.imread('C:\lab1\wp3770574-red-dead-redemption-2-4k-wallpapers.jpg', cv2.IMREAD_GRAYSCALE)

# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)

# Plot histograms
plt.figure(figsize=(12, 6))

# Original Image Histogram
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Histogram of Original Image')
plt.hist(image.ravel(), bins=256, range=[0, 256], color='black')

# CLAHE Image Histogram
plt.subplot(2, 2, 3)
plt.title('CLAHE Enhanced Image')
plt.imshow(clahe_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Histogram of CLAHE Enhanced Image')
plt.hist(clahe_image.ravel(), bins=256, range=[0, 256], color='black')

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




