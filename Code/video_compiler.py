import cv2
import os

# Define the directory where the images are located
img_dir = 'P3Data/renders/scene5'

# Define the video file name and codec
video_name = 'scene5.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Define the video frame size and FPS
frame_size = (1280, 960)
fps = 6

# Get the list of image filenames
img_files = sorted(os.listdir(img_dir))

# Create the video writer object
out = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

# Loop over the images and add them to the video writer object
for img_file in img_files:
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)
    out.write(img)

# Release the video writer object and destroy all windows
out.release()
cv2.destroyAllWindows()
