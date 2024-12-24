import os
import argparse
from datetime import datetime

from predict import Predictor

def get_outputs(src_image_fp: str,
                pose_list_dir: str,
                sample_algorithm: str = "ddim",
                num_poses: int = 1,
                nsteps: int = 50,
                output_dir: str = None):

    obj = Predictor(pose_list_dir)
    src_image_fp = os.path.join(os.getcwd(), src_image_fp)
    
    obj.predict_pose(image=src_image_fp,
                     sample_algorithm=sample_algorithm,
                     num_poses=num_poses,
                     nsteps=nsteps,
                     output_dir=output_dir
                    )

if __name__ == "__main__":
    
        # Get the current timestamp
    current_timestamp = datetime.now()
    # Format the timestamp in DDMMYY-HH:MM:SS format
    formatted_timestamp = current_timestamp.strftime("%d%m%y-%H:%M:%S")
    
    parser = argparse.ArgumentParser(description="Generate image using pre-trained PIDM.")

    parser.add_argument("src_image_fp", type=str, help="Filepath for source image.")
    parser.add_argument("pose_list_dir", type=str, default=None, help="Directory name for input poses .npy files.")
    parser.add_argument("--sample_algorithm", type=str, default="ddim", help="Type of algorithm to use for sampling.")
    parser.add_argument("--num_poses", type=int, default=1, help="Number of poses to generate for.")
    parser.add_argument("--nsteps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--output_dir", type=str, default=f"demo/output/{formatted_timestamp}", help='Name of output dir to save generated images to.')

    args = parser.parse_args()
    
    get_outputs(args.src_image_fp,
                args.pose_list_dir,
                args.sample_algorithm,
                args.num_poses,
                args.nsteps,
                args.output_dir)
    
    print(f"SUCCESS - Images successfully saved to {args.output_dir}.")


    '''
    Steps to get generated images from PIDM:
    - Get an image filepath from annotations csv and generate the npy file for the pose
    - Run the function here
    '''