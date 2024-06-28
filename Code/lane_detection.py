import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import csv
import os

# names = glob.glob('/home/blacksnow/Desktop/RBE549/msdiwan_p3/P3Data/Seq_Frames/interpolated_frames/scene4/*.jpg',recursive=True,)


def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)

    # Stretch the pixel values to cover the full dynamic range (0-255)
    stretched = np.uint8(255 * (gray - min_val) / (max_val - min_val))

    # Display the original and stretched images side by side
    # cv2.imshow('Original', gray)
    # cv2.imshow('Stretched', stretched)
    # cv2.waitKey(0)
    blur = cv2.GaussianBlur(stretched,(5,5),0)
    canny = cv2.Canny(blur, 0, 100)
    return canny

def roi(img):
    h =  img.shape[0]
    w = img.shape[1]
    # poly = np.array([[(50,h),(w, h),(w,475),(575,475)]])
    poly = np.array([[(0,(h//2+50)),(0,h),(w,h),(w,(h//2+50))]])

    mask = np.zeros_like(img)
    cv2.fillPoly(mask,poly,255)  
    masked_img  = cv2.bitwise_and(img,mask)
    return masked_img

def display_lines(img,lines):
    global x_list
    line_img = np.zeros_like(img)
    if lines is not None:
        x1_list = []
        x2_list = []
        # x_list = []
        for j,line in enumerate(lines):
            
            # print(j)
            # print(line)
            x1,y1,x2,y2 = line.reshape(4)
            
            # print('x1=',x1,'x2=',x2)
            if x1 == x2 or abs(y2 - y1) / abs(x2 - x1) < 0.5:
                # x1_list.append(1000)
                # x2_list.append(0)
                continue
            
            else:
                cv2.line(line_img,(x1,y1),(x2,y2),(255,0,0),10)
                x1_list.append(x1)
                x2_list.append(x2)
        
        # if len(x1_list) > 0:
        #     x_min = min(x1_list)
        #     # print(x_min)
        #     x_list.append(x_min)

        # if len(x2_list) > 0:
        #     x_max = max(x2_list)
        #     # print(x_max)
        #     x_list.append(x_max)

    return x1_list,x2_list,line_img
           
# x_list = []

# x = np.arange(0,2150,10)
# for i in x:
#     name = ('/home/blacksnow/Desktop/RBE549/msdiwan_p3/P3Data/Seq_Frames/interpolated_frames/scene7/')
#     img_name = ('frame'+str(i))
#     img = cv2.imread(name+img_name+'.jpg')
#     img_copy = np.copy(img)
#     rgb_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

#     canny_img  = canny(img_copy)
#     cropped_img = roi(canny_img)
#     lines = cv2.HoughLinesP(cropped_img,1,np.pi/180, 100 ,minLineLength=100,maxLineGap=20)
#     x1_list,x2_list,line_img = display_lines(img_copy,lines)
#     # print(len(x1_list))
#     # print(len(x2_list))

#     if len(x1_list) > 0:
#         x_min = min(x1_list)
#         # print(x_min)
#         # x_list.append(x_min)

#     if len(x2_list) > 0:
#         x_max = max(x2_list)
#         # print(x_max)
#         # x_list.append(x_max)
        
#     # print(x_min)
#     # print(x_max)
#     combo_img = cv2.addWeighted(img_copy,0.8,line_img, 1,1 )
#     cv2.imshow('result',combo_img)
#     cv2.waitKey(0)





#################################################################################################################################

img = cv2.imread("/home/blacksnow/Desktop/RBE549/msdiwan_p3/P3Data/Seq_Frames/interpolated_frames/scene4/frame85.jpg")
img_copy = np.copy(img)
rgb_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

canny_img  = canny(img_copy)
cropped_img = roi(canny_img)
lines = cv2.HoughLinesP(cropped_img,1,np.pi/180, 100 ,minLineLength=100,maxLineGap=20)
x1_list,x2_list,line_img = display_lines(img_copy,lines)
# print(len(x1_list))
# print(len(x2_list))

if len(x1_list) > 0:
    x_min = min(x1_list)
    # print(x_min)
    # x_list.append(x_min)

if len(x2_list) > 0:
    x_max = max(x2_list)
    # print(x_max)
    # x_list.append(x_max)
    
# print(x_min)
# print(x_max)
combo_img = cv2.addWeighted(img_copy,0.8,line_img, 1,1 )
cv2.imshow('result',combo_img)
cv2.waitKey(0)
cv2.imwrite('/home/blacksnow/Desktop/RBE549/msdiwan_p3/ppt_stuff/images/lanes/scene4frame85.jpg',combo_img)
##################################################################################################################################
    # csv_path = "./P3Data/final_csv_files/scene7/"
    # csv_name = csv_path+img_name+'.csv'

    # with open(csv_name) as csv_file:
    #     reader = csv.reader(csv_file, delimiter=',')
    #     line_count = 0
    #     data = list(reader)
    #     new_column = ['lanes_x']
    #     # print(data)

    #     for i,row in enumerate(data):
    #         # print('reading row'+str(i))
    #         if line_count == 0:
    #             line_count += 1
    #             continue
    #         else:
    #             new_column.append(str(x_min))
    #             new_column.append(str(x_max))

    #             # print(new_column)
    #             # plt.imshow(crop)
    #             # plt.show()
    #             line_count += 1

    #     for i, row in enumerate(data):
    #         row.append(new_column[i])


    #     # Save the new CSV file
    #     with open('./P3Data/final_csv_files/scene7/'+img_name+'.csv', 'w', newline='') as output_file:
    #         writer = csv.writer(output_file)
    #         writer.writerows(data)
#####################################################################################################################       


# # # Full video
# # cap = cv2.VideoCapture("./P3Data/Sequences/scene2/Undist/2023-03-03_10-31-11-front_undistort.mp4")
# # # cap = cv2.VideoCapture("./P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-front_undistort.mp4")
# # while(cap.isOpened()):
# #     _,frame = cap.read()
# #     canny_img  = canny(frame)
# #     cropped_img = roi(canny_img)
# #     lines = cv2.HoughLinesP(cropped_img,2,np.pi/180, 110 , np.array([]) ,minLineLength=20,maxLineGap=10)
# #     # print('lines=',lines)
# #     line_img = display_lines(frame,lines)
# #     combo_img = cv2.addWeighted(frame,0.8,line_img, 1,1) 
# #     cv2.imshow('result',combo_img)
# #     if cv2.waitKey(100) == ord('q'):
# #         break
# # cap.release()
# # cv2.destroyAllWindows()