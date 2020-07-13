import sys, os, argparse
import dlib
import glob
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F

import hopenet
import utils

DLIB_MODEL_PATH = "dlib_model/mmod_human_face_detector.dat"
HOPENET_MODEL_PATH = "hopenet_model/hopenet_alpha1.pkl"

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--mdlib', dest='face_model_dlib', help='Path of DLIB face detection model.',
          default=DLIB_MODEL_PATH, type=str)
    parser.add_argument('--mhopenet', dest='model_hopenet', help='Path of Hopenet model.',
          default=HOPENET_MODEL_PATH, type=str)
    parser.add_argument('--i', dest='input_image', help='Path of face image.', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.face_model_dlib):
        sys.exit('face_model_dlib is not exist!')

    if not os.path.exists(args.model_hopenet):
        sys.exit('model_hopenet is not exist!')
    
    if not os.path.exists(args.input_image):
        sys.exit('input_image is not exist!')

    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    gpu = 0
    batch_size = 1
    
    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Dlib face detection model
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_model_dlib)

    # Load hopenet snapshot
    model_hopenet_path = args.model_hopenet
    saved_state_dict = torch.load(model_hopenet_path)
    model.load_state_dict(saved_state_dict)

    transformations = transforms.Compose([transforms.Scale(224),
                            transforms.CenterCrop(224), transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    win = dlib.image_window()

    # Dlib face detector
    idx = 0
    for f in glob.glob(args.input_image + '/*.jpg'):
        print("Processing file: {}".format(f))
        
        img = dlib.load_rgb_image(f)
        cv_frame = cv2.imread(f)
        cvRGB_frame = cv2.cvtColor(cv_frame,cv2.COLOR_BGR2RGB)

        dets = cnn_face_detector(img, 1)

        print("Number of faces detected: {}".format(len(dets)))
        for i, det in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
                i, det.rect.left(), det.rect.top(), det.rect.right(), det.rect.bottom(), det.confidence))
            # Get x_min, y_min, x_max, y_max, conf
            x_min = det.rect.left()
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()
            conf = det.confidence

            # Plot expanded bounding box
            cv2.rectangle(cv_frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

            if conf > 1.0:
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                x_min -= int(2 * bbox_width / 4)
                x_max += int(2 * bbox_width / 4)
                y_min -= int(3 * bbox_height / 4)
                y_max += int(bbox_height / 4)
                x_min = max(x_min, 0); y_min = max(y_min, 0)
                x_max = min(cv_frame.shape[1], x_max); y_max = min(cv_frame.shape[0], y_max)

                # Crop image
                crop_img = cvRGB_frame[y_min:y_max,x_min:x_max]
                crop_img = Image.fromarray(crop_img)

                # Transform
                crop_img = transformations(crop_img)
                img_shape = crop_img.size()
                crop_img = crop_img.view(1, img_shape[0], img_shape[1], img_shape[2])
                crop_img = Variable(crop_img).cuda(gpu)

                yaw, pitch, roll = model(crop_img)

                yaw_predicted = F.softmax(yaw)
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)

                # Get continuous predictions in degrees.
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

                # Draw pose axis and cube
                cv_frame = utils.plot_pose_cube(cv_frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                cv_frame = utils.draw_axis(cv_frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
                
            

        fileName = "test" + str(idx) + ".jpg"
        cv2.imwrite(fileName, cv_frame)
        idx+=1