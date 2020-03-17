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
    for(size_t i =0, N = frame->mvpFeaturesLeft.size(); i < N; i++){
        auto mp = frame->mvpFeaturesLeft[i]->mpMapPoint.lock();
        if(mp != nullptr){
            mvpFeaturesLeft[i]->mpMapPoint = mp;
        }
    }

    _msePose = frame->Pose();

    // lack the process of setting KeyFrame->Feature->mnKF, which is done is CreateKF()
}

// ---------------------------------------------------------------------------------------------------------
KeyFrame::Ptr KeyFrame::CreateKF(Frame::Ptr frame){
    KeyFrame::Ptr newKF(new KeyFrame(frame));

    // link Feature->mpKF to the current KF
    // add the feature to Feature->MapPoint->observation
    for(auto &feat: newKF->mvpFeaturesLeft){
        feat->mpKF = newKF;
        auto mp = feat->mpMapPoint.lock();
        if(mp){
            mp->AddActiveObservation(feat);
        }
    }
    return newKF;
}

// -----------------------------------------------------------------------------------

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