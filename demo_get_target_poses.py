import os
import cv2
import json
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F

from PIL import Image
from data.fashion_base_function import get_transform

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(x_cords, -1), np.expand_dims(y_cords, -1)], axis=1)

def trans_keypoins(keypoints, param, img_size):
    missing_keypoint_index = keypoints == -1
    
    # NOTE: This was in original PIDM code but found not to be needed
    # NOTE: move all x coordinates to left by 40
    # keypoints[:,0] = (keypoints[:,0]-40)

    # resize the dataset
    # NOTE: This is becaue original openpose was meant to work with
    # images of (height, width) = (256,176)

    # Taken from CFLD: cords_to_map(array, (256, 256), (256, 176))
    img_h, img_w = img_size
    scale_w = 1.0/176.0 * img_w
    scale_h = 1.0/256.0 * img_h

    # NOTE: Scale to new image size
    if 'scale_size' in param and param['scale_size'] is not None:
        new_h, new_w = param['scale_size']
        scale_w = scale_w / img_w * new_w
        scale_h = scale_h / img_h * new_h
    
    # NOTE: Crop out specific parts
    if 'crop_param' in param and param['crop_param'] is not None:
        w, h, _, _ = param['crop_param']
    else:
        w, h = 0, 0

    keypoints[:,0] = keypoints[:,0]*scale_w - w
    keypoints[:,1] = keypoints[:,1]*scale_h - h
    keypoints[missing_keypoint_index] = -1
    return keypoints

def get_pose_np(img_filepath: str,
                annotations_filepath: str,
                output_dir: str):

    '''
    Takes in an input image and returns the desired np array of 20 channels.
    '''

    augment_params = {}

    # STEP: Load in reference image to generate image path from and transform it into tensor
    full_ref_img_path = os.path.join(os.getcwd(), img_filepath)
    ref_img_pil = Image.open(full_ref_img_path)

    # STEP: Resize to (256,256)
    ref_img_pil = ref_img_pil.resize((256,256))

    ref_img_transforms = get_transform(augment_params)
    ref_img = ref_img_transforms(ref_img_pil)

    # STEP: Get keypoint annotations for pose
    # NOTE: Store all the keypoints in a csv file
    annotation_file = pd.read_csv(os.path.join(os.getcwd(), annotations_filepath), sep=':')
    annotation_file = annotation_file.set_index('name')
    
    keypoint_string = annotation_file.loc[os.path.basename(img_filepath)]
    keypoint_array = load_pose_cords_from_strings(keypoint_string['keypoints_y'], keypoint_string['keypoints_x'])
    
    # STEP: Get pose map np array
    ref_img_np, face_center_np = get_label_tensor(keypoint_array, ref_img, augment_params)
    
    # STEP: Save to local file directory if output_dir provided
    if output_dir:
        output_dir = os.path.join(os.getcwd(), output_dir)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        img_file_name = os.path.basename(img_filepath).split(".")[0]
        saved_filepath = os.path.join(output_dir, f'{os.path.basename(img_file_name)}.npy')
        np.save(saved_filepath, ref_img_np)
        
        print(f"SUCCESS - Images successfully saved to {saved_filepath}.")

def get_label_tensor(keypoint, img, param):

    '''
    path: str, to keypoint annotations
    img: torch.tensor
    param: dict, param of rescaling and cropping, for data augmentation
    '''
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    canvas = np.zeros((img.shape[1], img.shape[2], 3)).astype(np.uint8)
    keypoint = trans_keypoins(keypoint, param, img.shape[1:])
    stickwidth = 4
    for i in range(18):
        x, y = keypoint[i, 0:2]
        if x == -1 or y == -1:
            continue
        cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    joints = []
    for i in range(17):
        Y = keypoint[np.array(limbSeq[i])-1, 0]
        X = keypoint[np.array(limbSeq[i])-1, 1]            
        cur_canvas = canvas.copy()
        if -1 in Y or -1 in X:
            joints.append(np.zeros_like(cur_canvas[:, :, 0]))
            continue
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        joint = np.zeros_like(cur_canvas[:, :, 0])
        cv2.fillConvexPoly(joint, polygon, 255)
        joint = cv2.addWeighted(joint, 0.4, joint, 0.6, 0)
        joints.append(joint)
    pose = F.to_tensor(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))

    tensors_dist = 0
    e = 1
    for i in range(len(joints)):
        im_dist = cv2.distanceTransform(255-joints[i], cv2.DIST_L1, 3)
        im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
        tensor_dist = F.to_tensor(Image.fromarray(im_dist))
        tensors_dist = tensor_dist if e == 1 else torch.cat([tensors_dist, tensor_dist])
        e += 1  

    label_tensor = torch.cat((pose, tensors_dist), dim=0)
    if int(keypoint[14, 0]) != -1 and int(keypoint[15, 0]) != -1:
        y0, x0 = keypoint[14, 0:2]
        y1, x1 = keypoint[15, 0:2]
        face_center = torch.tensor([y0, x0, y1, x1]).float()
    else:
        face_center = torch.tensor([-1, -1, -1, -1]).float()               
    return label_tensor, face_center

# TODO: Copy over the annotation csv file and  a test image to test it from there
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate pose map (numpy) for a reference image.")
    parser.add_argument("img_filepath", type=str, help="Path to reference image")
    parser.add_argument("annotations_filepath", type=str, help="Path to CSV file with keypoint annotations")
    parser.add_argument("output_dir", type=str, help="Path to output dir to store results")
    
    args = parser.parse_args()
    
    get_pose_np(args.img_filepath,
                args.annotations_filepath,
                args.output_dir)

    '''
    srun -p rtx3090_slab -n 1 --job-name=test-get-pose-pidm --kill-on-bad-exit=1 python3 demo_get_target_poses.py data/deepfashion_256x256/target_images/fashionMENJackets_Vestsid0000780001_4full.jpg  data/deepfashion_256x256/target_annotations/fasion-resize-annotation-test.csv data/deepfashion_256x256/target_pose 
    '''