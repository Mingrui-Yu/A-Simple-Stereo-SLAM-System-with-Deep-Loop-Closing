#include "myslam/frontend.h"

#include "myslam/feature.h"
#include "myslam/config.h"
#include "myslam/viewer.h"

namespace myslam{

// -------------------------------------------------------------
FrontEnd::FrontEnd(){
    _numFeatures = Config::Get<int>("num_features");

    _orb = cv::ORB::create(_numFeatures, 1.2f, 8, 31, 0, 2,
        cv::ORB::HARRIS_SCORE, 31, 20);

    _numFeaturesInit = Config::Get<int>("num_features_init");
    _numFeaturesTrackingGood = Config::Get<int>("num_features_tracking");
    _numFeaturesTrackingBad = Config::Get<int>("num_features_tracking_bad");
    _numFeaturesNeededForNewKF = Config::Get<int>("num_features_needed_for_keyframe");
}

// -------------------------------------------------------------
void FrontEnd::SetViewer(Viewer::Ptr viewer){
    _mpViewer = viewer;
}


// --------------------------------------------------------------
bool FrontEnd::GrabStereoImage (const cv::Mat &leftImg, const cv::Mat &rightImg, 
    const double &dTimeStamp){

    _mpCurrentFrame = Frame::Ptr(new Frame(leftImg, rightImg, dTimeStamp));

    switch(mStatus){
        case FrontEndStatus::INITING:
        StereoInit();
        break;
        case FrontEndStatus::TRACKING_GOOD:
        case FrontEndStatus::TRACKING_BAD:
            // Track();
            break;
        case FrontEndStatus::LOST:
            // Reset();
            break;
    }


    _mpViewer->AddCurrentFrame(_mpCurrentFrame);

    _mpLastFrame = _mpCurrentFrame;
    return true;
}


// --------------------------------------------------------------
bool FrontEnd::StereoInit(){
    DetectFeatures();

    return false;
}


// --------------------------------------------------------------
int FrontEnd::DetectFeatures(){
    cv::Mat mask(_mpCurrentFrame->mLeftImg.size(), CV_8UC1, 255);
    for (auto &feat: _mpCurrentFrame->mvpFeaturesLeft){
        cv::rectangle(mask, feat->mkpPosition.pt - cv::Point2f(10, 10),
            feat->mkpPosition.pt + cv::Point2f(10, 10), 0, CV_FILLED);
    }

    std::vector<cv::KeyPoint> keypoints;
    _orb->detect(_mpCurrentFrame->mLeftImg, keypoints, mask);
    int cntDetected = 0;
    for(auto &kp: keypoints) {
        _mpCurrentFrame->mvpFeaturesLeft.push_back(
            Feature::Ptr(new Feature(_mpCurrentFrame, kp)));
        cntDetected++;
    }
    LOG(INFO) << "Detect " << cntDetected << " new features in current frame";
    return cntDetected++;
}













} // namespace myslam