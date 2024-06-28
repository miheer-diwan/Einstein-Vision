import torch
import cv2
import numpy as np
import glob

# # Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# names = glob.glob('/home/blacksnow/Desktop/RBE549/msdiwan_p3/P3Data/Seq_Frames/interpolated_frames/scene2/*.jpg',recursive=True,)

x = np.arange(0,2200,10)

imgs=[]

for i in x:
    img_path = ('/home/blacksnow/Desktop/RBE549/msdiwan_p3/P3Data/Seq_Frames/interpolated_frames/scene8/')
    img_name = ('frame'+str(i))
    # print(img_name)
    # imgs.append(img_name)

# Inference
    results = model(img_path+img_name+'.jpg')
    # results = model(img_name)
    # Results
    results.print()
    results.save()  # or .show()

    results.xyxy[0]  # img1 predictions (tensor)
    # results = model(im)  # inference
    # crops = results.crop(save=True)  # cropped detections dictionary    
    x = results.pandas().xyxy[0]  # img1 predictions (pandas)
    x.to_csv('P3Data/csv_files/scene8/'+str(img_name)+ '.csv')

    print(results.pandas().xyxy[0])




