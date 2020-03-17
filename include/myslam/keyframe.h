#ifndef MYSLAM_KEYFRAME_H
#define MYSLAM_KEYFRAME_H

#include "myslam/common_include.h"

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

    SE3 Pose();


public:
    unsigned long mnFrameId;
    unsigned long mnKFId;
    double mdTimeStamp;

    std::vector<std::shared_ptr<Feature>> mvpFeaturesLeft;
    // std::vector<std::shared_ptr<Feature>> mvpFeaturesRight;

    


private:
    SE3 _msePose;

    std::mutex _mmutexPose;  // pose 数据锁
};




} // namespace myslam

#endif