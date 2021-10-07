# import the necessary packages
import numpy as np
#import argparse
import pandas as pd 
import time
import cv2
import os
from imutils import paths
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import backend as K

###################################################################################################################################################
#                                                    Step 1: Chop video into individual frames                                                    #
###################################################################################################################################################
activate_step_one = True

if activate_step_one:
    print("~~~~~ Running step 1: Processing video(s) ~~~~~")
    main_dir = os.getcwd()
    os.chdir(r'D:\Daniel\PMD\scripts\yoloV4PA\ffmpeg\bin') #change directory to .exe script
    cmd = r'python automate_cmd_ffmpeg.py'
    try:
        os.system(cmd)
    except:
        print("~~~ Failed to process video(s) ~~~")

    os.chdir(main_dir)
    print('#### Step 1 Completed: Processing Video(s) ####')




############################################################################################################################################################
#                                                    Step 2: label and crop images of different classes                                                    #
############################################################################################################################################################
#activate step 2a: detect person in frame and create XML file
activate_step_two_a = True

#activate step2b: crop detected person and classify and add cropped image into class folder
activate_step_two_b = True

#detector confidence and threshold settings
d_confidence = 0.50
d_threshold = 0.90

#classifier confidence setting
c_confidence = 0.50

####################
# Helper Functions #
####################
def xml_gen(file,path,b_width,b_height):
        f = open(( file.rsplit( ".", 1 )[ 0 ] ) + ".xml", "w") #creates a new file using the .jpg filename, but with the .fsv extension
        f.write("<annotation>") #write to the text file
        f.write("\n")  
        f.write("    <folder>" + os.path.basename(path) + "</folder>")  
        f.write("\n")        
        f.write("    <filename>" + file + "</filename>")    
        f.write("\n")        
        f.write("    <path>" + os.path.join(path,file) + "</path>")     
        f.write("\n")        
        f.write("    <source>\n     <database>Unknown</database>\n    </source>")   
        f.write("\n")  
        f.write("    <size>") 
        f.write("\n") 
        f.write("        <width>" + str(b_width) + "</width>")     
        f.write("\n")
        f.write("        <height>" + str(b_height) + "</height>")
        f.write("\n")  
        f.write("        <depth>3</depth>") 
        f.write("\n")  
        f.write("    </size>") 
        f.write("\n")        
        f.write("    <segmented>0</segmented>")     
        f.write("\n")
        

def obj_xml(file,label,xmin,ymin,xmax,ymax):
        f = open(( file.rsplit( ".", 1 )[ 0 ] ) + ".xml", "a") #creates a new file using the .jpg filename, but with the .fsv extension
        f.write("    <object>")
        f.write("\n")
        f.write("        <name>" + label + "</name>")
        f.write("\n")       
        f.write("        <pose>Unspecified</pose>\n        <truncated>0</truncated>\
        \n        <difficult>0</difficult>") 
        f.write("\n")        
        f.write("        <bndbox>")
        f.write("\n")   
        f.write("            <xmin>" + str(xmin) + "</xmin>") 
        f.write("\n")   
        f.write("            <ymin>" + str(ymin) + "</ymin>")   
        f.write("\n")        
        f.write("            <xmax>" + str(xmax) + "</xmax>")     
        f.write("\n")
        f.write("            <ymax>" + str(ymax) + "</ymax>")
        f.write("\n") 
        f.write("    </bndbox>")
        f.write("\n") 
        f.write("    </object>")
        f.write("\n") 


def close_xml(file):
        f = open(( file.rsplit( ".", 1 )[ 0 ] ) + ".xml", "a") #creates a new file using the .jpg filename, but with the .fsv extension
        f.write("</annotation>") 
        f.close()


def find_xml(fname, DATA_DIR):
    xml_name = fname[:-3]+'xml'
    xml_path = DATA_DIR+'/'+xml_name
    # print(xml_path)
    return xml_path


def extract_coords(xml_path):
    x_value = 1
    y_value = 1
    coords = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for object in root.findall('object'):
        name = object.find('name').text
        if(name == 'person'):
            bbox = object.find('bndbox')
            # for cords in bbox:
            xmin = int(int((bbox.find('xmin').text))/x_value) #1.02 - 1.03 for side views
            ymin = int(int((bbox.find('ymin').text))/y_value)
            xmax = int(int((bbox.find('xmax').text))*x_value)
            ymax = int(int((bbox.find('ymax').text))*y_value)
            coords.append([xmin,ymin,xmax,ymax])
    return coords


#creates class folders in a save directory
def create_class_folders(save_dir, classes):
    for clss in classes:
        try:
            os.mkdir(save_dir + '\\' + clss)
        except:
            print('unable to create class folder')


#locates h5 file in the h5_to_uff folder
def find_h5_file(folder):
    for filepath in glob.glob(folder + '\*'):
        filename = filepath.split('\\')[-1]
        if '.h5' in filename:
            return filename


#ensures image dimensions are not too small
def width_height_test(width, height, cam):
    if cam == 'cam1' or cam == 'cam2':
        if width>50 and height>50:
            return True
    elif cam == 'cam3':
        if width>65 and height>65:
            return True
    elif cam=='cam4':
        if width>55 and height>55:
            return True
    return False




##########################################################
# Initialise weights of YOLO Detector and PMD Classifier #
##########################################################
#################
# YOLO Detector #
#################
configPath = r'D:\Daniel\PMD\scripts\yoloV4PA\Yolov4-custom_new.cfg'
labelsPath = r'D:\Daniel\PMD\scripts\yoloV4PA\YoloV4PA.names'
weightsPath = r'D:\Daniel\PMD\scripts\yoloV4PA\yoloV4PA.weights'

LABELS = open(labelsPath).read().strip().split("\n")


print(weightsPath)
# load our YOLO object detector trained on OCR dataset (34 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]



##################
# PMD Classifier #
##################
# print("Setting Config")
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# session = tf.Session(config=config)
# K.set_session(session)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model_version = 47
weights_version = find_h5_file(r'D:\Daniel\PMD\scripts\H5 to UFF\pmd_v{}'.format(model_version))

print(f"Loading Model v{model_version}")

model_path = r"D:\Daniel\PMD\scripts\H5 to UFF\pmd_v{}\{}".format(model_version, weights_version)
model = load_model(model_path)

classes = ["cyclist", "ebike", 'motorcycle', 'pedestrian', 'standing_scooter']




###################################################################################
#                    Detect persons and classify cropped image                    #
###################################################################################
#folder containing 'day extracted' and 'night extracted' folders
DATA_FOLDER = r'I:\Daniel\Israel Footage\extracted images\extracted'

#save directory for cropped images
SAVE_DIR = r"I:\Daniel\Israel Footage\extracted images\persons" #night persons or day persons depending on time of video


###########################
# Step 2a: Detect persons #
###########################

if activate_step_two_a:

    print('step 2a: Detecting persons and creating XML files......')
    for extracted_folder in glob.glob(DATA_FOLDER + '\*'): #iterate through day & night extracted folders
        for video_folder in glob.glob(extracted_folder + '\*'):
            videoname = video_folder.split('\\')[-1]

            if videoname== 'finished':
                continue
            
            imagePaths = sorted(list(paths.list_images(f"{video_folder}")))

            for img in imagePaths:
                # load our input image and grab its spatial dimensions
                groundtruth = img.split('.jpg')[0]
                image = cv2.imread(img)
                (H, W) = image.shape[:2]


                # construct a blob from the input image and then perform a forward
                # pass of the YOLO object detector, giving us our bounding boxes and
                # associated probabilities
                temp_xml_file = groundtruth + ".xml"
                xml_gen(temp_xml_file,groundtruth,W,H) 
                blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (608, 608),swapRB=True, crop=False)
                net.setInput(blob)

                start = time.time()
                layerOutputs = net.forward(ln)
                end = time.time()

                # show timing information on YOLO
                print(groundtruth, "YOLO took {:.6f} seconds".format(end - start))

                # initialize our lists of detected bounding boxes, confidences, and
                # class IDs, respectively
                boxes = []
                confidences = []
                classIDs = []

                # loop over each of the layer outputs
                for output in layerOutputs:
                    # loop over each of the detections
                    for detection in output:
                        # extract the class ID and confidence (i.e., probability) of
                        # the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > d_confidence:
                            # scale the bounding box coordinates back relative to the
                            # size of the image, keeping in mind that YOLO actually
                            # returns the center (x, y)-coordinates of the bounding
                            # box followed by the boxes' width and height
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top and
                            # and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates, confidences,
                            # and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)
                            
                # apply non-maxima suppression to suppress weak, overlapping bounding
                # boxes
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, d_confidence,d_threshold)
                prediction_img = []
                rect_bbox = []
                # prediction_img = LABELS[classID]
                # ensure at least one detection exists
                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        prediction_img = LABELS[classIDs[i]]
                        rect_bbox.append([x,y,x+w,y+h,w,h,prediction_img])
                        # prediction_img.append(LABELS[classIDs[i]])
                        
                if len(rect_bbox)>0:
                    
                    for index_j, j in enumerate(rect_bbox):
                            for index_k, k in enumerate(rect_bbox):

                                if j!=k:
                                    j_bbox = j
                                    k_bbox = k
                                    if j_bbox[2] <= k_bbox[0] or j_bbox[3] <= k_bbox[1] or j_bbox[0] >= k_bbox[2] or  j_bbox[1] >= k_bbox[3]:
                                        overlap = True
                                    else:
                                        overlap = True
                                        if 'j_bbox' in globals() is True and 'k_bbox' in globals() is True:
                                            if j_bbox[6] == k_bbox[6] and j_bbox[2]<k_bbox[2]:
                                                rect_bbox.remove(k_bbox)
                                            elif j_bbox[6] == k_bbox[6] and j_bbox[2]>k_bbox[2]:
                                                rect_bbox.remove(j_bbox)


                for bbox in rect_bbox:
                        xmin = bbox[0] 
                        ymin = bbox[1] 
                        ymax = bbox[3] 
                        xmax = bbox[2]  
                        c = bbox[6]
                        if c =='person':
                            obj_xml(temp_xml_file, c,xmin,ymin,xmax,ymax)

                        
                        
                close_xml(temp_xml_file)





##########################################
# Step 2b: Crop Image and Classify Image #
##########################################
#image pre-processing
IMG_SIZE = (260,110)

#cam1 and cam2 crop size (Note: side view) : focus on adjusting width
cam1n2_xmin_cropsize = 1.04
cam1n2_xmax_cropsize = 1.04
cam1n2_ymin_cropsize = 1.03
cam1n2_ymax_cropsize = 1.03
#       NOTE: Ideal xmin & xmax cropsize
#               -> motorcycle: 1.05
#               -> cyclist: 1.04
#               -> standing_scooter: 1.03

#cam3 and cam4 crop size (Note: Front-back view) : focus on adjusting height
cam3n4_xmin_cropsize = 1.01
cam3n4_xmax_cropsize = 1.01
cam3n4_ymin_cropsize = 1.07
cam3n4_ymax_cropsize = 1.07


if activate_step_two_b:
    
    print('step 2b: Classify cropped image and copy into class folder......')
    count = 1
    for extracted_folder in glob.glob(DATA_FOLDER + '\*'): # 'day extracted' and 'night extracted'
        day_or_night = extracted_folder.split('\\')[-1]
        for video_dir in glob.glob(extracted_folder + '\*'): # iterate through each video folder of the extracted folders
            foldername = video_dir.split('\\')[-1]
            cam = foldername.split('-')[0]

            if foldername == 'finished':
                continue
        
            print(f"folder {count}: {foldername}")
            
            #create save directory for each video folder
            if day_or_night == 'day extracted':
                extracted_folder_name = 'day persons'
            elif day_or_night == 'night extracted':
                extracted_folder_name = 'night persons'

            save = SAVE_DIR + '\\' + extracted_folder_name + '\\' + foldername
            if os.path.exists(save):
                print('folder to save cropped images exist!')
                break
            os.mkdir(save)
            create_class_folders(save, classes)


            ###########################################
            #     Adjust cropsize for each camera     #
            ###########################################
            if (cam == 'cam1' or cam=='cam2'):
                xmin_cropsize = cam1n2_xmin_cropsize
                xmax_cropsize = cam1n2_xmax_cropsize
                ymin_cropsize = cam1n2_ymin_cropsize
                ymax_cropsize = cam1n2_ymax_cropsize
            
            elif (cam == 'cam3' or cam=='cam4'):
                xmin_cropsize = cam3n4_xmin_cropsize
                xmax_cropsize = cam3n4_xmax_cropsize
                ymin_cropsize = cam3n4_ymin_cropsize
                ymax_cropsize = cam3n4_ymax_cropsize           


            ##################################################
            #     iterating through one folder of frames     #
            ##################################################

            for fname in tqdm(os.listdir(video_dir)):
                if(r'.jpg' in fname):
                    try:
                        img = cv2.imread(os.path.join(video_dir, fname))

                        xml_path = find_xml(fname, video_dir)
                        coords = extract_coords(xml_path)

                        for (xmin, ymin, xmax, ymax) in coords:
                            if(xmax-xmin > 1 and ymax-ymin > 1):
                                xmin_new = int(xmin/xmin_cropsize)
                                xmax_new = int(xmax*xmax_cropsize)
                                ymin_new = int(ymin/ymin_cropsize)
                                ymax_new = int(ymax*ymax_cropsize)

                                width = xmax_new - xmin_new
                                height = ymax_new - ymin_new

                                #ensure image is not too small for each cam angle
                                if width_height_test(width, height, cam):
                                    
                                    #original cropped image (without adjusting crop size)
                                    original_cropped_img = img[ymin:ymax, xmin:xmax]


                                    ###################################
                                    #     Predict with Classifier     #
                                    ###################################
                                    
                                    #PREDICT CLASS with classifier: "cyclist", "ebike", 'motorcycle', 'pedestrian','seated_scooter', 'standing_scooter'
                                    cropped_img = img[ymin_new:ymax_new, xmin_new:xmax_new]
                                    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                                    cropped_img_array = cv2.resize(cropped_img, (IMG_SIZE[1], IMG_SIZE[0]), cv2.INTER_LANCZOS4) #classifier trained on (260,110) images
                                    cropped_img_array = cropped_img_array/255
                                    cropped_img_array = np.expand_dims(cropped_img_array, axis=0) #expand dimensions from 3 to 4 because tensorflow model takes in batch size as 1 additional dim

                                    prediction = model.predict(cropped_img_array)
                                    pred_class_index = np.argmax(prediction, axis = 1)
                                    pred_class = classes[pred_class_index[0]]
                                    
                                    
                                    #skip predictions that are below confidence value
                                    if prediction[0][np.argmax(prediction, axis=1)[0]] < c_confidence:
                                        continue

                                    #############################
                                    #     save cropped image    #
                                    #############################

                                    width_to_height_ratio = float(width/height)
                                    if cam=='cam1' or cam=='cam2':
                                        # if width_to_height_ratio <1 and pred_class=='pedestrian': #tall image for pedestrian
                                        #     save_crop = save + '\\' + pred_class 
                                        #     cv2.imwrite(os.path.join(save_crop, foldername +'_'+ fname[:-4]+'-'+str(len(os.listdir(SAVE_DIR)))+ ".jpg"), original_cropped_img)

                                        if width_to_height_ratio <1 and (pred_class=='standing_scooter' or pred_class=='pedestrian'): #tall image for standing_scooter and pedestrian
                                            save_crop = save + '\\' + pred_class 
                                            cv2.imwrite(os.path.join(save_crop, foldername +'_'+ fname[:-4]+'-'+str(len(os.listdir(SAVE_DIR)))+ ".jpg"), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

                                        elif pred_class!='pedestrian' or pred_class!='standing_scooter': #wide image for cyclist, ebike, motorcycle, seated_scooter
                                            save_crop = save + '\\' + pred_class 
                                            cv2.imwrite(os.path.join(save_crop, foldername +'_'+ fname[:-4]+'-'+str(len(os.listdir(SAVE_DIR)))+ ".jpg"), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
                                        

                                    elif cam=='cam3' or cam=='cam4':
                                        # if width_to_height_ratio <1 and pred_class=='pedestrian':
                                        #     save_crop = save + '\\' + pred_class 
                                        #     cv2.imwrite(os.path.join(save_crop, foldername +'_'+ fname[:-4]+'-'+str(len(os.listdir(SAVE_DIR)))+ ".jpg"), original_cropped_img)

                                        if width_to_height_ratio <1 and (pred_class=='standing_scooter' or pred_class=='pedestrian'): #tall image for standing_scooter and pedestrian
                                            save_crop = save + '\\' + pred_class 
                                            cv2.imwrite(os.path.join(save_crop, foldername +'_'+ fname[:-4]+'-'+str(len(os.listdir(SAVE_DIR)))+ ".jpg"), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

                                        elif pred_class != 'standing_scooter' or pred_class != 'pedestrian':
                                            save_crop = save + '\\' + pred_class 
                                            cv2.imwrite(os.path.join(save_crop, foldername +'_'+ fname[:-4]+'-'+str(len(os.listdir(SAVE_DIR)))+ ".jpg"), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
                                    
                                    

                    except Exception as e:
                        print(str(e)) #prints the error
                        pass
                #break
                
            count +=1
            
        #break


'''
	void ClassifierTensorRTMultiBatchFilter::enlargeRect(Rect &input_rect, double scale_factor_x, double scale_factor_y, double width, double height)
	{
		int newX = max(int(input_rect.x - input_rect.width * (scale_factor_x / 2)), 0);
		int newY = max(int(input_rect.y - input_rect.height * (scale_factor_y / 2)), 0);
		int newWidth = std::min(double(input_rect.width * (1 + scale_factor_x)), _width - newX);
		//int newWidth = int(input_rect.width * (1+scale_factor));
		//int newHeight = int(input_rect.height * (1+scale_factor));
		int newHeight = std::min(double(input_rect.height * (1 + scale_factor_y)), _height - newY);
		cv::Rect new_rect(newX, newY, newWidth, newHeight);
		input_rect = new_rect;
	}
'''