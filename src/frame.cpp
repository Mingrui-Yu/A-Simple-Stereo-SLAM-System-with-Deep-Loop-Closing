#include "myslam/frame.h"

namespace myslam{

unsigned long Frame::nNextId = 0;

// ---------------------------------------------------------------------------------------------------------
Frame::Frame(const cv::Mat &leftImg, const cv::Mat &rightImg, const double &dTimeStamp){
    mLeftImg = leftImg;
    mRightImg = rightImg;
    mdTimeStamp = dTimeStamp;

    mnFrameId = nNextId++;
}

// ---------------------------------------------------------------------------------------------------------
void Frame::SetPose(const SE3 &pose){
        std::unique_lock<std::mutex> lck(_mmutexPose);
        _msePose = pose;
    }

// ---------------------------------------------------------------------------------------------------------
SE3 Frame::GetPose() {
    std::unique_lock<std::mutex> lck(_mmutexPose);
    return _msePose;
}


}  // namespace myslam