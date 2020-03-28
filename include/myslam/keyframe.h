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

    std::vector<cv::KeyPoint> GetKeyPoints();


public:
    unsigned long mnFrameId;
    unsigned long mnKFId;
    double mdTimeStamp;

    std::weak_ptr<KeyFrame> mpLastKF;
    SE3 mRelativePoseToLastKF;
    std::weak_ptr<KeyFrame> mpLoopKF;
    SE3 mRelativePoseToLoopKF;

    std::vector<std::shared_ptr<Feature>> mvpFeaturesLeft;
    cv::Mat mImageLeft;
    DeepLCD::DescrVector mpDescrVector;
    // float mfSimilarityScoreNormFactor;
    // std::vector<std::shared_ptr<Feature>> mvpFeaturesRight;
    
    // std::vector<cv::KeyPoint> mvORBKpsLeft;
    cv::Mat mORBDescriptors;
    

private:
    

    SE3 _msePose;

    std::mutex _mmutexPose;  // pose 数据锁
};




} // namespace myslam

#endif