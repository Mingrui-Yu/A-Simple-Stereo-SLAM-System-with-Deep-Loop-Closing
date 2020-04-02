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

    // the relative pose to the reference KF
    void SetRelativePose(const SE3 &relativePose);

    SE3 Pose();

    SE3 RelativePose();


public:
    unsigned long mnFrameId;
    double mdTimeStamp;

    cv::Mat mLeftImg, mRightImg;

    std::vector<std::shared_ptr<Feature>> mvpFeaturesLeft;
    std::vector<std::shared_ptr<Feature>> mvpFeaturesRight;

    
private:
    SE3 _msePose; // just for viewer
    SE3 _mseRelativePose; // for tracking

    std::mutex _mmutexPose;
    std::mutex _mmutexRelativePose;


};




} // namespace myslam

#endif