#ifndef MYSLAM_KEYFRAME_H
#define MYSLAM_KEYFRAME_H

#include "myslam/common_include.h"
#include "myslam/deeplcd.h"

namespace myslam{

// forward declaration
class Feature;
class Frame;


class KeyFrame{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<KeyFrame> Ptr;

    KeyFrame() {}

    KeyFrame(std::shared_ptr<Frame> frame);

    static KeyFrame::Ptr CreateKF(std::shared_ptr<Frame> frame);

    void SetPose(const SE3 &pose);

    SE3 Pose(); // Tcw

    /**
     * get all features' keypoints (not pyramid)
     */
    std::vector<cv::KeyPoint> GetKeyPoints();


public:
    unsigned long mnFrameId;
    unsigned long mnKFId;
    double mdTimeStamp;
    std::vector<std::shared_ptr<Feature>> mvpFeaturesLeft;
    cv::Mat mImageLeft;
    
    // for pose graph optimization
    std::weak_ptr<KeyFrame> mpLastKF;
    SE3 mRelativePoseToLastKF;
    std::weak_ptr<KeyFrame> mpLoopKF;
    SE3 mRelativePoseToLoopKF;

     // pyramid keypoints only for computing ORB descriptors and doing matching
    std::vector<cv::KeyPoint> mvPyramidKeyPoints;
    
    DeepLCD::DescrVector mpDescrVector;
    cv::Mat mORBDescriptors;
    

private:
    
    SE3 _msePose;

    std::mutex _mmutexPose;
};




} // namespace myslam

#endif