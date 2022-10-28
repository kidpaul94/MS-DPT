# Run following commands in Google Colab or local Linux environment
# Downgrade torch to accommodate DCNv2
!pip install torchvision==0.5.0
!pip install torch==1.4.0

# Make sure we're in the top folder
%cd /content/drive/MyDrive/

# Clone the repo
!git clone https://github.com/dbolya/yolact.git

# Change to the right directory
%cd /content/drive/MyDrive/yolact/external/DCNv2

# Build DCNv2
!python setup.py build develop

# Make sure we're in the top folder
%cd /content/drive/MyDrive/

!python ./yolact/train.py --config=yolact_plus_resnet50_config --resume=/content/drive/MyDrive/weights/yolact_plus_resnet50_54_800000.pth --start_iter=0

# Move up to the top level directory
%cd /content/drive/MyDrive/

# Run inference using our pre-trained weights on all images in the directory
!python ./yolact/eval.py --trained_model=./weights/yolact_plus_resnet50_10_10000.pth --config=yolact_plus_resnet50_config --score_threshold=0.90 --top_k=15 --images=test_images:output_images 
# --display_bbox=False --display_text=False

# Simple python script to show output images.
import cv2
import cupy as np
from matplotlib import pyplot as plt
from pathlib import Path

output_images = Path('output_images')

def show_image(img_path):
  img = cv2.imread(img_path)
  img_cvt=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.figure(figsize=(16,16))
  plt.imshow(img_cvt)
  plt.show()

# Iterate through all of the output images and display them
for img_path in output_images.iterdir():
  print(img_path)
  show_image(str(img_path))
