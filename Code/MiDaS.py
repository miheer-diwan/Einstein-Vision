import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import csv
import torch
    

# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)

x = np.arange(1740,2155,10)

for i in x:
    print(i)
    # torch.cuda.empty_cache()

    # free_gpu_cache()        

    # img_path = ('/home/blacksnow/Desktop/RBE549/msdiwan_p3/P3Data/Seq_Frames/interpolated_frames/scene5/')
    img_name = ('frame'+str(i))

    filename = ('/home/blacksnow/Desktop/RBE549/msdiwan_p3/P3Data/Seq_Frames/interpolated_frames/scene7/'+img_name+'.jpg')

    # # Load a model (see https://github.com/intel-isl/MiDaS/#Accuracy for an overview)
    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # # Move model to GPU if available
    device = torch.device("cpu") 
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # # Load transforms to resize and normalize the image for large or small model
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()


    # # output approach 1
    output = prediction.cpu().numpy()

    ## verifying shape of ip adn op images
    # print('img shape =', np.shape(img))
    # print('output shape =', np.shape(output))

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(img)
    # # plt.show()

    # plt.subplot(1,2,2)
    # plt.imshow(output,cmap ='gray')
    # plt.show()

    max = np.max(output)-np.min(output)
    min = np.min(output)-np.min(output)

    # print('max=',max)

    ## Read CSV file
    csv_path = "P3Data/csv_files/scene7/"
    csv_name = csv_path+img_name+'.csv'

    with open(csv_name) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        data = list(reader)
        new_column = ['depth']
        # print(data)

        for i,row in enumerate(data):
            # print('reading row'+str(i))
            if line_count == 0:
                line_count += 1
                continue
            else:
                crop = output[int(float(row[2])):int(float(row[4])),int(float(row[1])):int(float(row[3]))]

                # print('crop depth=',np.average(crop))
                depth = max - np.average(crop)
                # print('scaled depth=',depth)
                new_column.append(str(depth))
                # print(new_column)
                # plt.imshow(crop)
                # plt.show()
                line_count += 1

        for i, row in enumerate(data):
            row.append(new_column[i])


        # Save the new CSV file
        with open('P3Data/final_csv_files/scene7/'+img_name+'.csv', 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerows(data)


    print(f'Processed {line_count} lines.')
    # torch.cuda.empty_cache()

