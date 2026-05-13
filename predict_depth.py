import torch
from depth_anything_3.api import DepthAnything3

import matplotlib.pyplot as plt
import numpy as np
import cv2

from pathlib import Path
from PIL import Image

# Load model from Hugging Face Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/da3nested-giant-large")
model = model.to(device=device)

# Run inference on images
image_dir = Path("/home/lab605/SLAM_3DGS/Track2/data/image") # List of image paths, PIL Images, or numpy arrays
output_dir = Path("/home/lab605/SLAM_3DGS/Track2/npz_out")
output_dir.mkdir(exist_ok=True)

images = sorted([str(p) for p in image_dir.glob("*.jpg")])
print(f"found {len(images)} images...")

batch_size = 8

for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    prediction = model.inference(batch)
    
    for j, img_path in enumerate(batch):
        img_name = Path(img_path).stem
        orig_img = Image.open(img_path)  
        
        #  Resize 
        depth = np.array(Image.fromarray(prediction.depth[j]).resize(orig_img.size, Image.Resampling.BILINEAR))
        conf = np.array(Image.fromarray(prediction.conf[j]).resize(orig_img.size, Image.BILINEAR))
        
        np.savez_compressed(
            output_dir / f"{img_name}.npz",
            depth=depth.astype(np.float32),
            conf=conf.astype(np.float32)
        )
    
    torch.cuda.empty_cache()
    
print("finished !")

"""
# Access results
print(prediction.depth.shape)        # Depth maps: [N, H, W] float32

depth = prediction.depth[0]
conf = prediction.conf[0]
# Normalize depth for visualization
vmin = np.percentile(depth, 2)
vmax = np.percentile(depth, 98)

depth_vis = np.clip((depth - vmin) / (vmax - vmin), 0, 1)

# Apply colormap
depth_vis_color = cv2.applyColorMap(
    (depth_vis * 255).astype(np.uint8),
    cv2.COLORMAP_INFERNO
)

image = cv2.imread(images[0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("RGB")
plt.imshow(image)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Depth")
plt.imshow(depth_vis, cmap='inferno')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Confidence")
plt.imshow(conf, cmap='viridis')
plt.axis("off")

plt.tight_layout()
plt.show()
"""
