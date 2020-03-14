#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "myslam/common_include.h"

namespace myslam{

// forward declaration
class Feature;


class Frame{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    Frame() {}

    Frame(const cv::Mat &leftImg, const cv::Mat &rightImg, const double &dTimeStamp);

    void SetPose(const SE3 &pose);

    SE3 GetPose();

public:
    unsigned long mnFrameId;
    unsigned long mnKFid;
    static unsigned long nNextId;
    double mdTimeStamp;

    cv::Mat mLeftImg, mRightImg;

    std::vector<std::shared_ptr<Feature>> mvpFeaturesLeft;
    std::vector<std::shared_ptr<Feature>> mvpFeaturesRight;

    


private:
    SE3 _msePose;

    std::mutex _mmutexPose;  // pose 数据锁


};




} // namespace myslam

#endif