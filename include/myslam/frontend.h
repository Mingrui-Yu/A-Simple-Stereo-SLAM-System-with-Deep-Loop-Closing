#ifndef MYSLAM_FRONTEND_H
#define MYSLAM_FRONTEND_H

#include "myslam/common_include.h"
#include "myslam/frame.h"

#include <opencv2/features2d.hpp>

namespace myslam{

class Viewer;
class ORBextractor;
class Camera;
class Map;
class Backend;
class KeyFrame;

enum class FrontendStatus {INITING, TRACKING_GOOD, TRACKING_BAD, LOST};

class Frontend{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frontend> Ptr;

    Frontend();

    // process new pair of images
    bool GrabStereoImage(const cv::Mat &leftImg, const cv::Mat &rightImg, const double &dTimeStamp);

    void SetViewer(std::shared_ptr<Viewer> viewer){
        _mpViewer = viewer;
    }

    // set left and right camera
    void SetCameras(std::shared_ptr<Camera> left, std::shared_ptr<Camera> right){
        _mpCameraLeft = left;
        _mpCameraRight = right;
    }

    // set the map
    void SetMap(std::shared_ptr<Map> map){
        _mpMap = map;
    }

    void SetBackend(std::shared_ptr<Backend> backend){
        _mpBackend = backend;
    }

    void SetORBextractor(std::shared_ptr<ORBextractor> orb){
        _mpORBextractor = orb;
    }

    FrontendStatus GetStatus() const {
        return _mStatus;
    }


    

private:
     // tracking initialization 
    bool StereoInit();

    // tracking, return true if success
    bool Track();

    // return if has built the map successfully
    bool BuildInitMap();

    // detect features in an image, return the num of features
    int DetectFeatures();

    // find the corresponding features in right image of current frame
    // return num of corresponding features found
    int FindFeaturesInRight();

    // get the initial value of current frame's pose using motion model
    // correspond features between last frame and current frame  using LK flow
    // return the number of good tracked points
    int TrackLastFrame();

    // optimize the current frame's pose using g2o
    int EstimateCurrentPose();

    // create new keyframe (from current frame) when the number of tracked points is less than a threshold
    // return if success
    bool InsertKeyFrame();

    // create new mappoints and add them to the map
    int TriangulateNewPoints();

    

private:
    FrontendStatus _mStatus = FrontendStatus::INITING;

    Frame::Ptr _mpCurrentFrame;
    Frame::Ptr _mpLastFrame;
    std::shared_ptr<KeyFrame> _mpReferenceKF;


    SE3 _mseRelativeMotion;
    SE3 _mseRelativeMotionToReferenceKF;

    // params for tracking features
    int _numFeaturesTrackingGood;
    int _numFeaturesTrackingBad;
    int _numFeaturesInitGood;
    int _numFeaturesTracking;
    // cv::Ptr<cv::ORB> _orb;
    // cv::Ptr<cv::GFTTDetector> _gftt;
    std::shared_ptr<ORBextractor> _mpORBextractor, _mpORBextractorInit;

    std::shared_ptr<Camera> _mpCameraLeft, _mpCameraRight;

    std::shared_ptr<Map> _mpMap;

    std::shared_ptr<Backend> _mpBackend;

    // Other thread Pointers
    std::shared_ptr<Viewer> _mpViewer;

   
};




} // namespace myslam

#endif