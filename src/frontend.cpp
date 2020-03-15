#include "myslam/frontend.h"

#include "myslam/feature.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"
#include "myslam/config.h"
#include "myslam/viewer.h"
#include "myslam/ORBextractor.h"
#include "myslam/camera.h"

namespace myslam{

// -------------------------------------------------------------
FrontEnd::FrontEnd(){
    _numFeatures = Config::Get<int>("num_features");
    _numFeaturesInit = Config::Get<int>("num_features_init");
    _numFeaturesTrackingGood = Config::Get<int>("num_features_tracking");
    _numFeaturesTrackingBad = Config::Get<int>("num_features_tracking_bad");
    _numFeaturesNeededForNewKF = Config::Get<int>("num_features_needed_for_keyframe");

    // int nFeatures =  Config::Get<int>("ORBextractor.nFeatures");
    float fScaleFactor = Config::Get<float>("ORBextractor.scaleFactor");
    int nLevels = Config::Get<int>("ORBextractor.nLevels");
    int fIniThFAST = Config::Get<int>("ORBextractor.iniThFAST");
    int fMinThFAST = Config::Get<int>("ORBextractor.minThFAST");

    // _orb = cv::ORB::create(_numFeatures, 1.2f, 8, 31, 0, 2,
    //     cv::ORB::HARRIS_SCORE, 31, 20);
    // _gftt = cv::GFTTDetector::create(_numFeatures, 0.01, 20);

    mpORBextractor = ORBextractor::Ptr(new ORBextractor(_numFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST));
}

// -------------------------------------------------------------
void FrontEnd::SetViewer(Viewer::Ptr viewer){
    _mpViewer = viewer;
}


// --------------------------------------------------------------
bool FrontEnd::GrabStereoImage (const cv::Mat &leftImg, const cv::Mat &rightImg, 
    const double &dTimeStamp){

    _mpCurrentFrame = Frame::Ptr(new Frame(leftImg, rightImg, dTimeStamp));

    // LOG(INFO) << "frame id: " << _mpCurrentFrame->mnFrameId;

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
    int numCoorFeatures = FindFeaturesInRight();
    if (numCoorFeatures < _numFeaturesInit){
        return false;
    }

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
    cv::Mat descriptors;
    mpORBextractor->Detect(_mpCurrentFrame->mLeftImg, mask, keypoints);
    int cntDetected = 0;
    for(auto &kp: keypoints) {
        _mpCurrentFrame->mvpFeaturesLeft.push_back(
            Feature::Ptr(new Feature(_mpCurrentFrame, kp)));
        cntDetected++;
    }
    LOG(INFO) << "Detect " << cntDetected << " new features in current frame";
    return cntDetected++;
}




// --------------------------------------------------------------
int FrontEnd::FindFeaturesInRight(){
    // use LK flow to estimate points in the right image

    // set the initial postion of features in right frame
    std::vector<cv::Point2f> vpointLeftKPs, vpointRightKPs;
    vpointLeftKPs.reserve(_mpCurrentFrame->mvpFeaturesLeft.size());
    vpointRightKPs.reserve(_mpCurrentFrame->mvpFeaturesLeft.size());
    for(auto &kp:  _mpCurrentFrame->mvpFeaturesLeft){
        vpointLeftKPs.push_back(kp->mkpPosition.pt);
        auto mp = kp->mpMapPoint.lock();
        if(mp){
            // use projected points as initial guess
            Vec2 px = _cameraRight->world2pixel(mp->Pos(), _mpCurrentFrame->Pose());
            vpointRightKPs.push_back(cv::Point2f(px[0], px[1]));
        }else{
            // use same pixel in left image
            vpointRightKPs.push_back(kp->mkpPosition.pt);
        }
    }

    // LK flow: calculate the keypoints' positions in right frame
    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK( _mpCurrentFrame->mLeftImg,
        _mpCurrentFrame->mRightImg, vpointLeftKPs, vpointRightKPs, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // create new feature objects in right frame
    int numGoodPoints = 0;
    _mpCurrentFrame->mvpFeaturesRight.reserve(status.size());
    for(size_t i = 0, N = status.size(); i < N; ++i){
        if(status[i]){
            cv::KeyPoint kp(vpointRightKPs[i], 7);
            Feature::Ptr feat(new Feature(_mpCurrentFrame, kp));
            feat->mbIsOnLeftFrame = false;
            _mpCurrentFrame->mvpFeaturesRight.push_back(feat);
            numGoodPoints++;
        } else{
            _mpCurrentFrame->mvpFeaturesRight.push_back(nullptr);
        }
    }

    LOG(INFO) << "Find " << numGoodPoints << " corresponding features in the right image.";
    return numGoodPoints;
}













} // namespace myslam