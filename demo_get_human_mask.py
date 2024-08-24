import os
import argparse
import torch

from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights


def load_model(model_name: str,
               num_classes: int = 21):

    if model_name not in ("resnet_50", "resnet_101"):
        raise ValueError("Model name provided is invalid. Please use resnet_50 or resnet_101.")

    if model_name == "resnet_101":
        model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT,
                                    num_classes=num_classes
                                    )
        transforms = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

    else:
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT,
                                   num_classes=num_classes)
        transforms = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.transforms()
    # set to inference mode
    model.eval()
    return model, transforms

def segment_image(src_image_fp: str,
                  model_name: str,
                  output_dir: str):

    full_src_image_fp = os.path.join(os.getcwd(), src_image_fp)
    src_image_pil = Image.open(full_src_image_fp).convert("RGB")

    model, transforms = load_model(model_name)
    transformed_img = torch.unsqueeze(transforms(src_image_pil), 0)
    model_output = model(transformed_img)
    
    preds = model_output['out'][0].argmax(0)
    preds_np = preds.numpy()

    # class label of 15 for person in VOC and 1 for person in COCO
    # TODO: 0 for background, 15 for human
    preds_np[preds_np == 0] = 255
    preds_np[preds_np == 15] = 0
    mask_image = Image.fromarray(preds_np.astype("uint8"))

    if output_dir:
        output_dir = os.path.join(os.getcwd(), output_dir)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        input_fn = os.path.basename(full_src_image_fp)
        input_fn = input_fn.split(".")[0]
        print(input_fn)
        input_img_no = input_fn.split("_")[-1]
        print(input_img_no)
        mask_image.save(os.path.join(output_dir, f"ref_mask_{input_img_no}.jpg"))
        print(f"SUCCESS - Mask saved successfully at {output_dir}.")


if __name__ == "__main__":
    

    # STEP: Get mask for source image
    parser = argparse.ArgumentParser(description="Generate image using pre-trained PIDM.")

    parser.add_argument("src_image_fp", type=str, help="File path for source image.")
    parser.add_argument("model_name", type=str, default="resnet_50", help="Backbone model for segmentation.")
    parser.add_argument("output_dir", type=str, help="File path to save output.")
    args = parser.parse_args()

    segment_image(args.src_image_fp,
                  args.model_name,
                  args.output_dir)
