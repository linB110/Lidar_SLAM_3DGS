import os
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from tqdm import tqdm

class SkySegmenter:
    def __init__(self, device="cuda"):
        self.device = device
        model_id = "nvidia/segformer-b5-finetuned-ade-640-640"
        print(f"[Init] Loading {model_id}...")
        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
        self.model.eval()

    @torch.no_grad()
    def process_image(self, image_path, output_path):
        image = Image.open(image_path).convert("RGB")
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        # Inference
        outputs = self.model(**inputs)
        logits = outputs.logits  # [1, 150, H/4, W/4]
        
        # Resize to original image size
        upscaled_logits = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        
        # ADE20K label index 2 is "Sky"
        prediction = torch.argmax(upscaled_logits, dim=1).squeeze(0).cpu().numpy()
        mask = (prediction == 2).astype(np.uint8) * 255
        
        # Save binary mask
        cv2.imwrite(output_path, mask)

def main():
    # --- EDIT THESE PATHS ---
    track_dir = "/home/lab605/SLAM_3DGS/Track2"
    # Prefer pre-undistorted images if available
    _undist = f"{track_dir}/data/image_undistorted"
    input_dir = _undist if os.path.isdir(_undist) else f"{track_dir}/data/image"
    output_dir = f"{track_dir}/result/sky_masks"
    # -------------------------

    if not os.path.exists(input_dir):
        print(f"[Error] Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[Info] Found {len(img_files)} images in {input_dir}")

    if len(img_files) == 0:
        print("[Error] No images found. Check your 'input_dir' path.")
        return

    segmenter = SkySegmenter()

    for img_name in tqdm(img_files, desc="Processing Sky Masks"):
        in_path = os.path.join(input_dir, img_name)
        # Save as .png to avoid compression artifacts
        out_name = os.path.splitext(img_name)[0] + ".png"
        out_path = os.path.join(output_dir, out_name)
        segmenter.process_image(in_path, out_path)
    
    print(f"[Done] Masks saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
