# Training for YOLACT++
 _Transfer Learning (Training Soft-max Classification Layer ONLY)_

 •	https://github.com/dbolya/yolact/issues/334

 •	https://github.com/dbolya/yolact/issues/436 

 _Fine Tuning_

 •	https://github.com/dbolya/yolact/issues/36

 •	https://github.com/dbolya/yolact/issues/618

 •	https://github.com/dbolya/yolact/issues/454 -> Just see how he modifies config.py but not anything else.

 These are just reference I used to do fine tuning. You can read these if you want to understand more about the process. Every part that was 
 modified for transfer learning are already saved in Gitlab. More detail about training process (e.g., training time) can be found in the final 
 report as well. Note that you may not need 8000 ~ 9000 image data for a single class transfer learning. I presume that 1000 ~ 2000 are enough. 
 You can also find more information about proper number of images for transfer learning at here as well:

 •	https://github.com/dbolya/yolact/issues 

# Google Colab

![alt text](<./google-colab.png>) 

 Google Colab is a notebook that provides free GPU computation. You can mount Google Colab with your Google Drive to store and train the DNN. 
 **Google_colab.py** contains the list of commands you need to proceed for either transfer learning or fine tuning.  

 _How to use_
 
 •	https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d

# More information for training with custom dataset

 •	https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

