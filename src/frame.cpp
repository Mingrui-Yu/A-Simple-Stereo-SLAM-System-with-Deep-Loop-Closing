#include "myslam/frame.h"
#include "myslam/config.h"

namespace myslam{

// Frame::nFactoryId = 0;

// ---------------------------------------------------------------------------------------------------------
Frame::Frame(const cv::Mat &leftImg, const cv::Mat &rightImg, const double &dTimeStamp){
    static unsigned long nFactoryId = 0;

    mLeftImg = leftImg;
    mRightImg = rightImg;
    mdTimeStamp = dTimeStamp;

    mnFrameId = nFactoryId++;
}

// ---------------------------------------------------------------------------------------------------------
void Frame::SetPose(const SE3 &pose){
        std::unique_lock<std::mutex> lck(_mmutexPose);
        _msePose = pose;
    }


// ---------------------------------------------------------------------------------------------------------
void Frame::SetRelativePose(const SE3 &relativePose){
        std::unique_lock<std::mutex> lck(_mmutexRelativePose);
        _mseRelativePose = relativePose;
    }

// ---------------------------------------------------------------------------------------------------------
SE3 Frame::Pose() {
    std::unique_lock<std::mutex> lck(_mmutexPose);
    return _msePose;
}


// ---------------------------------------------------------------------------------------------------------
SE3 Frame::RelativePose() {
    std::unique_lock<std::mutex> lck(_mmutexRelativePose);
    return _mseRelativePose;
}

}  // namespace myslam