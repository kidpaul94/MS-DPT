# DNN - ICP
  Object 6d pose estimation using an RGB-D camera.

  Project under [Barton Research Group](https://brg.engin.umich.edu/), via [Multidisciplinary-Design-Program (MDP)](https://mdp.engin.umich.edu/)
  at University of Michigan (UMich).

# Components from Other Repositories
  YOLACT++ (https://github.com/dbolya/yolact) for object detections and segmentation.

    @article{yolact-plus-tpami2020,
      author  = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
      journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
      title   = {YOLACT++: Better Real-time Instance Segmentation}, 
      year    = {2020},  
    }

# How to run
  1) Run DNN - ICP pose estimation
```
cd ./dnn_pose/src/yolact/
python3 pose_estimation.py --config=yolact_plus_resnet50_config --score_threshold=0.90
```
