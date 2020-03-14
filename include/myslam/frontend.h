#ifndef MYSLAM_FRONTEND_H
#define MYSLAM_FRONTEND_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/viewer.h"

#include <opencv2/features2d.hpp>

namespace myslam{

class Viewer;

enum class FrontEndStatus {INITING, TRACKING_GOOD, TRACKING_BAD, LOST};

class FrontEnd{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<FrontEnd> Ptr;

    FrontEnd();

    void SetViewer(Viewer::Ptr viewer);

    // process new pair of images
    bool GrabStereoImage(const cv::Mat &leftImg, const cv::Mat &rightImg, const double &dTimeStamp);

     // tracking initialization 
    bool StereoInit();

    // detect features in an image, return the num of features
    int DetectFeatures();

public: 

    
    FrontEndStatus mStatus = FrontEndStatus::INITING;

private:
    Frame::Ptr _mpCurrentFrame;
    Frame::Ptr _mpLastFrame;

    // params for tracking features
    int _numFeatures;
    int _numFeaturesInit;
    int _numFeaturesTrackingGood;
    int _numFeaturesTrackingBad;
    int _numFeaturesNeededForNewKF;

    // Other thread Pointers
    Viewer::Ptr _mpViewer;

    cv::Ptr<cv::ORB> _orb;
};




} // namespace myslam

#endif