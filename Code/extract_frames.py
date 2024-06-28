# Program To Read video
# and Extract Frames

import cv2

# Function to extract frames
def FrameCapture(path):

	# Path to video file
	vidObj = cv2.VideoCapture(path)

	# Used as counter variable
	count = 0

	# checks whether frames were extracted
	success = 1

	while success:

		# vidObj object calls read
		# function extract frames
		success, image = vidObj.read()

		# Saves the frames with frame-count
		cv2.imwrite("./P3Data/Seq_Frames/all_frames/scene9/frame%d.jpg" % count, image)

		count += 1


# Driver Code
if __name__ == '__main__':

	# Calling the function
	# FrameCapture("P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-front_undistort.mp4")
	# FrameCapture("P3Data/Sequences/scene2/Undist/2023-03-03_10-31-11-front_undistort.mp4")
	# FrameCapture("P3Data/Sequences/scene3/Undist/2023-02-14_11-49-54-front_undistort.mp4")
	# FrameCapture("P3Data/Sequences/scene4/Undist/2023-02-14_11-51-54-front_undistort.mp4")
	# FrameCapture("P3Data/Sequences/scene5/Undist/2023-02-14_11-56-56-front_undistort.mp4")
	# FrameCapture("P3Data/Sequences/scene6/Undist/2023-03-03_15-31-56-front_undistort.mp4")
	# FrameCapture("P3Data/Sequences/scene7/Undist/2023-03-03_11-21-43-front_undistort.mp4")
	# FrameCapture("P3Data/Sequences/scene8/Undist/2023-03-03_11-40-47-front_undistort.mp4")
	FrameCapture("P3Data/Sequences/scene9/Undist/2023-03-04_17-20-36-front_undistort.mp4")
