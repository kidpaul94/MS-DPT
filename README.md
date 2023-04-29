<p align="center">
<img src=./images/logo.png width=40% height=40%>
</p>

# DNNPOSE

Online object 6D pose estimation for Human Support Robot([HSR](https://mag.toyota.co.uk/toyota-human-support-robot/)) from Toyota Research Institute.

![Example 0](./images/MS-DPT.png)

Project under Barton Research Group ([BRG](https://brg.engin.umich.edu/)), via Multidisciplinary-Design-Program ([MDP](https://mdp.engin.umich.edu/)) at the University of Michigan. The repository only provides a pipeline for ***6D pose estimation using an RGB-D image*** at the moment.

<!-- <p align="center">
<img src=./images/Log2.gif width=30% height=30%> <img src=./images/Log3.gif width=30% height=30%>
</p> -->

## Table of Contents

- [Repository Structure](#repository-structure)
- [Download Process](#download-process)
- [How to Run](#how-to-run)
    - [DNNPOSE](#dnnpose)
    - [Sensor Fusion](#sensor-fusion)
- [Citation](#citation)
- [ToDo Lists](#todo-lists)

---

## Repository Structure

    ├── dnnpose
    │   ├── launch             # ROS launch file
    │   └── src
    │       ├── yolact   
    │       ├── dnnpose_utils                      
    │       └── estimation.py  # Python code for OPE                    
    └── images
    
## Download Process

    cd ~/catkin_ws/src
    git clone https://github.com/kidpaul94/MS-DPT.git
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash

## How to Run

### DNNPOSE:

    cd dnnpose/src/
    rosrun estimation.py --trained_model=your_weights.pth --config=yolact_plus_resnet50_config --score_threshold=0.90

### Sensor Fusion:

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Anton&size=29&pause=1000&color=F70000&width=435&lines=TO+BE+CONTINUE" alt="Typing SVG" />
</a>
</br>

## Citation

    @article{lee2022multi,
      title={Multi-sensor aided deep pose tracking},
      author={Lee, Hojun and Toner, Tyler and Tilbury, Dawn and Barton, Kira},
      journal={IFAC-PapersOnLine},
      volume={55},
      number={37},
      pages={326--332},
      year={2022},
      publisher={Elsevier}
    }

## ToDo Lists

| **Refactorization** | ![Progress](https://progress-bar.dev/50) |
| --- | --- |
| **Multi-sensor fusion** | ![Progress](https://progress-bar.dev/0) |
