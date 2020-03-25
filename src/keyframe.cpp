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
    mImageLeft = frame->mLeftImg.clone();

    mvpFeaturesLeft = frame->mvpFeaturesLeft;
    // mvpFeaturesRight = frame->mvpFeaturesRight;
    for(size_t i =0, N = frame->mvpFeaturesLeft.size(); i < N; i++){
        auto mp = frame->mvpFeaturesLeft[i]->mpMapPoint.lock();
        if(mp != nullptr){
            mvpFeaturesLeft[i]->mpMapPoint = mp;
        }
    }
    
    // lack the process of setting KeyFrame->Feature->mnKF, which is done is CreateKF()
}

// ---------------------------------------------------------------------------------------------------------
KeyFrame::Ptr KeyFrame::CreateKF(Frame::Ptr frame){
    KeyFrame::Ptr newKF(new KeyFrame(frame));

    // link Feature->mpKF to the current KF
    // add the feature to Feature->MapPoint->observation
    for(size_t i = 0, N =  newKF->mvpFeaturesLeft.size(); i < N; i++){
        auto feat = newKF->mvpFeaturesLeft[i];
        feat->mpKF = newKF;
        // auto mp = feat->mpMapPoint.lock();
        // if(mp){
        //     mp->AddActiveObservation(feat);
        // }
    }


    return newKF;
}

// -----------------------------------------------------------------------------------

std::vector<cv::KeyPoint> KeyFrame::GetKeyPoints(){
    std::vector<cv::KeyPoint> vKeyPoints(mvpFeaturesLeft.size());
    for(size_t i = 0, N = mvpFeaturesLeft.size(); i < N; i++){
        vKeyPoints[i] = mvpFeaturesLeft[i]->mkpPosition;
    }
    return vKeyPoints;
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