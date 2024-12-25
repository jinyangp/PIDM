import math
import json
import argparse
import warnings
warnings.filterwarnings('ignore')
import torch
import torchvision.transforms.functional as F
from PIL import Image
from pathlib import Path
from datetime import datetime
from tensorfn import load_config as DiffConfig
import pandas as pd
import numpy as np
from config.diffconfig import DiffusionConfig, get_model_conf
import os, glob, cv2, time, shutil
from data.fashion_base_function import get_transform
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
import torchvision.transforms as transforms

def convert_fname(x):
    a, b = os.path.split(x)
    i = b.rfind('_')
    x = a + '/' +b[:i] + b[i+1:]
    return 'fashion'+x.split('.jpg')[0].replace('id_','id').replace('/','')

def get_name(src, dst):
    src = convert_fname(src)
    dst = convert_fname(dst)
    return src + '___' + dst

class Predictor():
    def __init__(self,
                 annotations_filepath="data/deepfashion_256x256/target_annotations/fasion-resize-annotation-test.csv"):
        
        """Load the model into memory to make running multiple predictions efficient"""

        conf = DiffConfig(DiffusionConfig, './config/diffusion.conf', show=False)

        self.model = get_model_conf().make_model()
        ckpt = torch.load("checkpoints/last.pt")
        self.model.load_state_dict(ckpt["ema"])
        self.model = self.model.cuda()
        self.model.eval()

        self.betas = conf.diffusion.beta_schedule.make()
        self.diffusion = create_gaussian_diffusion(self.betas, predict_xstart = False)#.to(device)
        
        # self.pose_list = glob.glob(pose_list_dir + "/*.npy")
        self.image_root =  os.path.join(os.getcwd(), "data", "imgs")
        os.makedirs(self.image_root, exist_ok=True)
        self.annotation_file = pd.read_csv(os.path.join(os.getcwd(), annotations_filepath), sep=':')
        self.annotation_file = self.annotation_file.set_index('name')

        self.output_root_dir = os.path.join(os.getcwd(), "demo", "outputs")

        self.transforms = transforms.Compose([transforms.Resize((256,256), interpolation=Image.BICUBIC),
                            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])
        self.INPUT_WIDTH = 256
        self.INPUT_HEIGHT = 256

    def load_pose_cords_from_strings(self, y_str, x_str):
        
        y_cords = json.loads(y_str)
        x_cords = json.loads(x_str)
        return np.concatenate([np.expand_dims(x_cords, -1), np.expand_dims(y_cords, -1)], axis=1)

    def trans_keypoins(self, keypoints, param, img_size):
        
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
    
    def get_label_tensor(self, keypoint, img, param):

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
        keypoint = self.trans_keypoins(keypoint, param, img.shape[1:])
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

    def get_pose_np(self,
                    image_fp,
                    image):
        
        augment_params = {}

        image = image.resize((256,256))
        ref_img_transforms = get_transform(augment_params)
        image = ref_img_transforms(image)

        keypoint_string = self.annotation_file.loc[(os.path.basename(image_fp) + ".jpg")]
        keypoint_array = self.load_pose_cords_from_strings(keypoint_string['keypoints_y'], keypoint_string['keypoints_x'])
        
        # STEP: Get pose map np array
        img_tensor, face_center_tensor = self.get_label_tensor(keypoint_array, image, augment_params)

        return img_tensor

    def resize_img(self,
                   img: Image.Image,
                   new_width: int,
                   new_height: int) -> Image.Image:
        return img.resize((new_width, new_height))

    def predictions_from_csv(self,
                            data_csv_fp,
                            project_name,
                            sample_algorithm='ddim',
                            nsteps=100
                            ):
        
        # STEP: Read in the csv file
        data_csv_full_fp = os.path.join(os.getcwd(), data_csv_fp)
        data_csv_df = pd.read_csv(data_csv_full_fp)

        # STEP: Settle input/output directories
        image_root = Path(self.image_root)

        output_dir = os.path.join(self.output_root_dir, project_name)
        samples_dir = os.path.join(output_dir, "samples")
        concat_dir = os.path.join(output_dir, "concat")
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(concat_dir, exist_ok=True)
        f_ext = "png"

        # STEP: Iterate through the data rows
        for idx, row in data_csv_df.iterrows():

            # STEP: Read in src and target image as PIL images
            fname = get_name(row['from'], row['to'])
            src_img_fp = str(image_root/row['from'])
            target_img_fp = str(image_root/row['to'])

            src = Image.open(src_img_fp)
            src = self.transforms(src).unsqueeze(0).cuda()
            
            # STEP: Get pose.npy file from the target image
            target = Image.open(target_img_fp)
            converted_target_fn = convert_fname(row['to'])
            target_tensor = self.get_pose_np(converted_target_fn, target)
            target_pose = torch.stack([target_tensor.cuda()], 0)

            if sample_algorithm == 'ddpm':
                samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, target_pose], progress = True, cond_scale = 2)
            elif sample_algorithm == 'ddim':
                noise = torch.randn(src.shape).cuda()
                seq = range(0, 1000, 1000//nsteps)
                xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, target_pose])
                samples = xs[-1].cuda()

            # clamps range to [-1,1] and scales it to [0,1]
            # (1,3,256,256)
            samples = (torch.clamp(samples, -1., 1.) + 1.0)/2.0

            # STEP: Save the sample
            samples = samples.squeeze(0)
            samples_pil = transforms.ToPILImage()(samples)
            samples_pil.save(os.path.join(samples_dir, f'{fname}.{f_ext}'))

            src_pil = Image.open(src_img_fp)
            src_pil = self.resize_img(src_pil, self.INPUT_WIDTH, self.INPUT_HEIGHT)
            src_tensor = torch.tensor(np.array(src_pil)).permute(2,0,1)
            
            target_pil = Image.open(target_img_fp)
            target_pil = self.resize_img(target_pil, self.INPUT_WIDTH, self.INPUT_HEIGHT)
            target_tensor = torch.tensor(np.array(target_pil)).permute(2,0,1)
            
            samples_pil = self.resize_img(samples_pil, self.INPUT_WIDTH, self.INPUT_HEIGHT)
            samples_tensor = torch.tensor(np.array(samples_pil)).permute(2,0,1)

            # STEP: Save the concat result
            concat = transforms.Resize([256, 528])(torch.cat([src_tensor.detach().cpu(),
                                                     target_tensor.detach().cpu(),
                                                     samples_tensor.detach().cpu()], 2))
            transforms.ToPILImage()(concat).save(os.path.join(concat_dir, f'{fname}.{f_ext}'))

if __name__ == "__main__":

    # STEP: Get parameters
    # Get the current timestamp
    current_timestamp = datetime.now()
    # Format the timestamp in DDMMYY-HH:MM:SS format
    formatted_timestamp = current_timestamp.strftime("%d%m%y-%H:%M:%S")

    parser = argparse.ArgumentParser(description="Description of your program.")
    parser.add_argument("data_csv_fp", type=str, help="Data csv file path containing the file paths of style and pose image for each sample.")
    parser.add_argument("project_name", type=str, default=formatted_timestamp, help="Folder name to store outputs under.")
    parser.add_argument("--sample_algorithm", type=str, default="ddim", help="Sampling algorithm to use.")
    parser.add_argument("--nsteps", type=int, default=100, help="Number of sampling steps in sampling algorithm.")
    args = parser.parse_args()

    obj = Predictor()
    obj.predictions_from_csv(args.data_csv_fp,
                             args.project_name,
                             sample_algorithm=args.sample_algorithm,
                             nsteps=args.nsteps)
    
    '''
    srun -p rtx3090_slab -n 1 --job-name=test --gres=gpu:1 --kill-on-bad-exit=1 python3 -u demo_batch_predict_pose.py data/benchmark-test-pairs.csv 251224-benchmark-test --sample_algorithm=ddim --nsteps=100
    '''