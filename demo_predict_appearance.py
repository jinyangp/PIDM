import os
import argparse
from datetime import datetime
import numpy as np

from PIL import Image

from predict import Predictor

def get_outputs(src_image_fp: str,
                ref_image_fp: str,
                ref_mask_fp: str,
                ref_pose_fp: str,
                sample_algorithm: str = "ddim",
                nsteps: int = 100,
                output_dir: str = None):
    
    '''
    The idea here is that:
    1. Source image: Contains the background style we want to have in the generated image.
    --> Use a source image that contains just the background
    2. Reference image: Contains the image that has the subject we want to have in our generated image.
    --> Since do not have a pre-trained UNET with the weights required,
    --> we will use an image from DeepFashion dataset as ref image
    --> we take segmented map of that ref image and segment the bg from the fg
    3. Reference mask: Contains the mask that seperates the background and foreground.
    --> https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html
    --> DeepLabV3_ResNet101_Weights.DEFAULT (use this for weights parameter)
    4. Reference pose: Contains the pose we want our subject to be in.
    --> but, how do we get pose in npy form?
    '''

    obj = Predictor()

    full_src_image_fp = os.path.join(os.getcwd(), src_image_fp)
    full_ref_image_fp = os.path.join(os.getcwd(), ref_image_fp)
    full_ref_mask_fp = os.path.join(os.getcwd(), ref_mask_fp)
    full_ref_pose_fp = os.path.join(os.getcwd(), ref_pose_fp)
 
    obj.predict_appearance(full_src_image_fp,
                           full_ref_image_fp,
                           full_ref_mask_fp,
                           full_ref_pose_fp,
                           output_dir=output_dir)


if __name__ == "__main__":
    
    # STEP: Get prediction for appearance
    # Get the current timestamp
    current_timestamp = datetime.now()
    # Format the timestamp in DDMMYY-HH:MM:SS format
    formatted_timestamp = current_timestamp.strftime("%d%m%y-%H:%M:%S")
    
    parser = argparse.ArgumentParser(description="Generate image using pre-trained PIDM.")

    parser.add_argument("src_image_fp", type=str, help="Filepath for source image.")
    parser.add_argument("ref_image_fp", type=str, help="Filepath for ref image.")
    parser.add_argument("ref_mask_fp", type=str, help="Filepath for ref mask.")
    parser.add_argument("ref_pose_fp", type=str, help="Filepath for ref pose.")
    parser.add_argument("--sample_algorithm", type=str, default="ddim", help="Type of algorithm to use for sampling.")
    parser.add_argument("--nsteps", type=int, default=100, help="Number of inference steps.")
    parser.add_argument("--output_dir", type=str, default=f"demo/output/{formatted_timestamp}", help='Name of output dir to save generated images to.')

    args = parser.parse_args()
    
    get_outputs(args.src_image_fp,
                args.ref_image_fp,
                args.ref_mask_fp,
                args.ref_pose_fp,
                args.sample_algorithm,
                args.nsteps,
                args.output_dir)
    
    print(f"SUCCESS - Images successfully saved to {args.output_dir}.")