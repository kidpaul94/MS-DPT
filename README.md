# DNN - ICP 6d Pose Estimation
  Object 6d pose estimation for the [HSR](https://robots.ieee.org/robots/hsr/).

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

  robot_localization (https://github.com/cra-ros-pkg/robot_localization) for Extended Kalman Filter.

# How to run
  1) Run DNN - ICP pose estimation
```
cd ./dnn_pose/src/yolact/
python3 pose_estimation.py --config=yolact_plus_resnet50_config --score_threshold=0.90
```
  2) Run odometry transformation node
```
cd ./dnn_pose/src/yolact/
python3 base2object.py
```
  3) Run EKF node 
```
roslaunch robot_localization ekf_template.launch
```
