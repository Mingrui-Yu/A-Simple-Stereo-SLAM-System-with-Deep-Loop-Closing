#ifndef MYSLAM_FRONTEND_H
#define MYSLAM_FRONTEND_H

#include "myslam/common_include.h"
#include "myslam/frame.h"

#include <opencv2/features2d.hpp>

namespace myslam{

class Viewer;
class ORBextractor;
class Camera;

enum class FrontEndStatus {INITING, TRACKING_GOOD, TRACKING_BAD, LOST};

class FrontEnd{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<FrontEnd> Ptr;

    FrontEnd();

    void SetViewer(std::shared_ptr<Viewer> viewer);

    // process new pair of images
    bool GrabStereoImage(const cv::Mat &leftImg, const cv::Mat &rightImg, const double &dTimeStamp);

     // tracking initialization 
    bool StereoInit();

    // detect features in an image, return the num of features
    int DetectFeatures();

    // find the corresponding features in right image of current frame
    // return num of corresponding features found
    int FindFeaturesInRight();

    // set left and right camera
    void SetCameras(std::shared_ptr<Camera> left, std::shared_ptr<Camera> right){
        _cameraLeft = left;
         _cameraRight = right;
    }

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
    // cv::Ptr<cv::ORB> _orb;
    // cv::Ptr<cv::GFTTDetector> _gftt;
    std::shared_ptr<ORBextractor> mpORBextractor;

    // Other thread Pointers
    std::shared_ptr<Viewer> _mpViewer;

    std::shared_ptr<Camera> _cameraLeft, _cameraRight;
};




} // namespace myslam

#endif