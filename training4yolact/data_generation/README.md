# Notice
  This is written based on **example.blend** I created with Blender. Blender is an open 3D computer graphics software whose GUI can be controlled
  by Python script. Such characteristic of the software allows its user to easily automate some processes. This document majorly covers features of 
  Blender related to generating synthetic dataset for YOLACT++. This will help you to understand overall path of dataset generation, and possibly 
  give some idea for improving this process. Therefore, I recommend that you explore example.blend while checking each section of the document.

# Synthetic Dataset
  Training Deep neural network such as YOLACT++ requires great amount of manually labeled image data for each class of object we want to detect and 
  segment. To reduce such pain in preparing dataset for training, I decided to synthesize images using CAD files and make python script which does 
  labeling process for me. Following flowchart shows how this is done:

  ![alt text](<./data_prep.jpg>) 

  You can get more description about this flowchart in my final report. I tried to automate the data preparation process as much as possible within 
  limited time frame. Currently, Python script I created for Blender (blender.py) can automatically rotate objects, change their colors, and save 
  both RGB and grayscale images. Contour detection and labelme2coco are fully automated to generate labeling as well. However, there are still some
  parts of this process that needs to be manually controlled such as background image, object texture image, blur, and material 
  properties. RGB and grayscale images are also be sorted out if objects are seriously occluded or truncated.     

# Custom COCO dataset for YOLACT++ 
  We will use Blender_obj_random.py to generate synthetic images semi-automatically. You can download Blender and put the python   
  script (Blender_obj_random.py) in the Blender. More detail will be provided with an example blender file (example.blend). 
  After generating all images, run Annotation_gen.py. This will generate labelme format annotation.json file for all RGB images 
  using the coressponding grayscale images. Finally, run label2coco so that we can get final custom COCO dataset for training. label2coco requires
  custom labels in .txt files (custom_labels.txt). You should make both training dataset and validation dataset to train YOLACT++.

  1) Blender_obj_random.py
  Blender_obj_random.py is a script for automatically controlling pose and color of objects in a scene. Currently, this script will 
  automatically generates 100 of images with those variations and saves as RGB AND GRAYSCALE images.

  2) Annotation_gen.py
  Annotation_gen.py is a script that automatically generates "labelme" annotation format of objects in each RGB image using 
  corresponding grayscale image. Generated labelme format can be converted to COCO dataset using labelme2coco.

  3) Labelme2coco
  labelme2coco is a conversion file from labelme annotation format to COCO dataset format (https://github.com/wkentaro/labelme)

    @misc{labelme2016,
      author =       {Kentaro Wada},
      title =        {{labelme: Image Polygonal Annotation with Python}},
      howpublished = {\url{https://github.com/wkentaro/labelme}},
      year =         {2016}
    }

# Basic operations of Blender
  In the following section, I will provide some basic information so that you can familiarize yourself with Blender:

  _Move / Navigate around your scene_

  •	http://www.youtube.com/watch?v=K6Sm7DAPTGE

  _Import STL file (CAD)_

  •	Click “File” on the top left  Click “Import”  Select the STL file
 
  _Move / Rotate / Scale objects_

  •	http://www.youtube.com/watch?v=0QYrQOLEWAo
  
  _Import background images_

  •	https://www.youtube.com/watch?v=LNujGgsB2VI

  •	Note that I put Scale block in example.blend unlike the video above. This allows us to control how the image fits in the camera frame (Stretch  
  is used).

  _Import object texture images_

  •	https://www.youtube.com/watch?v=r5YNJghc81U

  •	https://www.youtube.com/watch?v=XI-pZshRp8g

  •	You can also see more detail in example.blend by 1) select object in the collection tree (orange highlight) and 2) click Shading and Texture
  Paint. 
  
  _Generate grayscale images_

  •	https://www.youtube.com/watch?v=xeprI8hJAH8&t=589s

  •	Checkout example.blend as well.  

# Real Image Data
  Synthetic data is great because we can quickly generate more various dataset, which may increase generalizability of a neural network.
  Nontheless, we use the network with real image data for object detection. So, it will be a good idea to add real images from the robot camera to 
  validation dataset. This will be helpful to evaluate whether the trained network works well with the real images. If the network trained with 
  synthetic dataset does not perform well, we can consider to mix the synthetic images with the real images for perparing a new training dataset. **Also, background images (images without any object) must be included in dataset to reduce FP(False Positive) cases.**
