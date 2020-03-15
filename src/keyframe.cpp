#include "myslam/keyframe.h"

#include "myslam/frame.h"
#include "myslam/feature.h"
#include "myslam/mappoint.h"


namespace myslam{

KeyFrame::KeyFrame(Frame::Ptr frame){
    static unsigned long nFactoryId = 0;

    mnKFId = nFactoryId++;
    mnFrameId = frame->mnFrameId;
    mdTimeStamp = frame->mdTimeStamp;

    mvpFeaturesLeft = frame->mvpFeaturesLeft;
    // mvpFeaturesRight = frame->mvpFeaturesRight;

    _msePose = frame->Pose();
}


// ---------------------------------------------------------------------------------------------------------
void KeyFrame::SetPose(const SE3 &pose){
        std::unique_lock<std::mutex> lck(_mmutexPose);
        _msePose = pose;
    }

// ---------------------------------------------------------------------------------------------------------
SE3 KeyFrame::Pose() {
    std::unique_lock<std::mutex> lck(_mmutexPose);
    return _msePose;
}

















} // namespace myslam