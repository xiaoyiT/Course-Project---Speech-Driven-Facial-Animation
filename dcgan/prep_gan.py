import cv2
import numpy as np
import dlib
import os
from pydub import AudioSegment
from scipy.fftpack import fft

detector = dlib.get_frontal_face_detector() #Face detector
predictor= dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file

def find_landmarks_from_frame(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)

    detections = detector(clahe_image, 1) #Detect the faces in the image

    for k,d in enumerate(detections): #For each detected face
        landmarks = [] #construct a list containing all coordinates of landmarks
        shape = predictor(clahe_image, d) #Get coordinates
        
        for i in range(0,68): #There are 68 landmark points on each face
            landmarks.append((shape.part(i).x, shape.part(i).y))
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
    
    return landmarks


def get_normalization_standard_points(landmarks):
    landmarks_array = np.array(landmarks)
    xmax = np.max(landmarks_array[:,0])
    xmin = np.min(landmarks_array[:,0])
    ymax = np.max(landmarks_array[:,1])
    ymin = np.min(landmarks_array[:,1])
    
    return {"xmax":xmax,"xmin":xmin,"ymax":ymax,"ymin":ymin}
    
def normalize_landmarks(landmarks,standard_points):
    normalized_landmarks = []
    x_length =  standard_points['xmax'] -  standard_points['xmin']
    y_length =  standard_points['ymax'] -  standard_points['ymin']
    
    for pair in landmarks:
        normalized_x = (pair[0] - standard_points['xmin']) / float(x_length)
        normalized_y = (pair[1] - standard_points['ymin']) / float(y_length)
        normalized_landmarks.extend((normalized_x,normalized_y))
   
    return normalized_landmarks

all_landmarks =[]
all_fft = []
file_count=1
files = os.listdir('./video')
files = sorted(files)
for filename in files:
    print("Preprocess Video "+str(file_count))
    file_count+=1
    video_landmarks = []
    if filename != ".DS_Store":
        vidcap = cv2.VideoCapture('./video/'+filename)
        success,image = vidcap.read()
        count = 0
        step=30
        success = True
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,count*step)
            success,image = vidcap.read()
            if success:
		if count%10 == 0:      
			cv2.imwrite('./data/fr/'+filename.replace('.mp4','')+'_'+'0'*(4-len(str(count)))+str(count)+'.jpg',image)
                	#count += 1
                	landmarks = find_landmarks_from_frame(image)
                	standard_points = get_normalization_standard_points(landmarks)
                	normalized_landmarks = normalize_landmarks(landmarks,standard_points)
                	all_landmarks.append(np.array(normalized_landmarks).T.tolist())
		count += 1
        


np.save('./data/landmarks.npy',np.array(all_landmarks,dtype=np.float32))
#np.save('./train/train_fft.npy',np.asarray(all_fft,dtype=np.float32))
