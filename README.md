# A Simple Stereo SLAM System with Deep Loop Closing

[中文传送门](https://www.cnblogs.com/MingruiYu/p/12634631.html)

This is a simple stereo SLAM system with a deep-learning based loop closing module. As a beginner of SLAM, I made this system mainly in order to practice my coding and engineering skills to build a full SLAM system by myself. 

I chose to build this system based on stereo cameras because it is easiler, without complicated work on initialization or dealing with the unknown scale. The structure of the system is simple and clear, in which I didn't apply much detailed optimization. Thus, the performance of this system is not outstanding. However, I hold the view that such a simple structure may be friendly for a SLAM beginner to study the body frame of a full SLAM system. It will be definitely a tough work for a beginner to study, for example,  ORB-SLAM2, a complex system with more than 10 thousand lines of code and a lot of tricks to improve its performance. 

It is truly a pleasure for me if this project can help you. 

## Related References

### Chapter 13, Visual SLAM: From Theory to Practice
(https://github.com/gaoxiang12/slambook-en), I use the basic framework of the Stereo VO in the chapter, and the main methods in the frontend and the backend threads.

### ORB-SLAM2
(https://github.com/raulmur/ORB_SLAM2), I use some modified code in ORB-SLAM2 (mainly from ORBextractor.cpp for ORB feature extracting). 

### Lightweight Unsupervised Deep Loop Closure
(https://github.com/rpng/calc), I use modified versions of the DeepLCD library to perform loop detection.

## Dependencies

The platform I use is Ubuntu 18.04.

### OpenCV

Dowload and install instructions can be found at [here](https://opencv.org/releases/). I am using OpenCV 3.4.8.

### Eigen

Dowload and install instructions can be found at [here](https://opencv.org/releases/). You can just use 
``` sudo apt-get install libeigen3-dev ```
to install it in Ubuntu.

### g2o
Dowload and install instructions can be found at [here](https://github.com/RainerKuemmerle/g2o).

### Caffe
The CPU verion is enough. Dowload and install instructions can be found at [here](http://caffe.berkeleyvision.org/install_apt.html). This is the dependency of DeepLCD library.

NOTICE: Please make sure your caffe is installed in ~/caffe, or you need to change its path in CMakeLists.txt.

### DeepLCD
This is the library for deep loop detection. The modified library are included in the project, so you don't need to install it by yourself.

### Others
* C++11
* Boost filesystem
* Google Logging Library (glog)

TIPS: There may be some dependencies I missed. Please open an issue if you face any problem.

## Build

Clone the repository:
```
git clone https://github.com/Mingrui-Yu/A-Simple-Stereo-SLAM-System-with-Deep-Loop-Closing
```

Build:
```
mkdir build
cd build
cmake ..
make
```

This will create libmyslam.so in /lib folder and the executables in /bin folder.

## Run the example

Up to now I have just written the example to run KITTI Stereo. The main file is at /app/run_kitti_stereo.

First, download a sequence from [KITTI Database]( http://vision.in.tum.de/data/datasets/rgbd-dataset/download), or you can download it from [Baidu Netdisk link](https://www.sohu.com/a/219232053_715754) shared by Pao Pao Robot in China.

To run the system on KITTI Stereo sequence 00:
```
./bin/run_kitti_stereo  config/stereo/gray/KITTI00-02.yaml  PATH_TO_DATASET_FOLDER/dataset/sequences/00
```
where KITTI00-02.yaml is the corresponding configuration file (including camera parameters and other parameters). It utilizes the style of ORB-SLAM2. 

Besides, some parameters in configuration file are for viewing:
* Camera.fps: control the frame rate of the system
* LoopClosing.bShowResult: whether to show the match result and reprojection result in Loop Closing
* Viewer.bShow: whether to show the frame and map in real time while system running

Here is a result of keyframe trajectory in KITTI 00.

<div align=center><img src="https://img2020.cnblogs.com/blog/1921421/202004/1921421-20200404204617560-1899206987.png" width = "60%" /></div>

The system can run at a frame rate of around 50 frames per second (if the viewer is closed). If you don't need to undistort the images (such as in KITTI database), it can even accelerate to around 100 frames per second. (Run on a laptop with  i5-8265U(1.60GHz × 8) and no GPU)

# Brief Introduction

The system contains three thread:
* Frontend thread
* Backend thread
* LoopClosing thread

In Frontend, it will track the motion based on feature points and LK flow. If the number of tracked keypoints is lower than a thresold, it will detect new features and create a keyframe. Mappoints are created by triangulating the matched feature points in left/right images.

In Backend, it will maintain a global map and an local active map. The active map is like a sliding window, containing a fixed number of keyframes and observed mappoints. Optimization of the active map is done in Backend.

In LoopClosing, it will first try to detect a Candidate Loop KF of the Current KF using DeepLCD. If succeed, it will then match the keypoints in Candidate KF and Current KF, which is used to compute the correct pose of Current KF using PnP and g2o optimization. If the number of inliers is higher than a threshold, the loop detection will be considered as a success, and loop correction is applyed: first, it will correct the keyframe poses and mappoint positions in active map; second, a pose graph optimization of the global map will be applied.


***

There must be some mistakes in the project as I am just a newcomer to visual SLAM. Please open an issue if you find any problem, and I will be deeply grateful for your correction and advice.







