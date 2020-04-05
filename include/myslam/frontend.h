#ifndef MYSLAM_FRONTEND_H
#define MYSLAM_FRONTEND_H

#include "myslam/common_include.h"

#include <opencv2/features2d.hpp>

namespace myslam{

class Frame;
class Viewer;
class ORBextractor;
class Camera;
class Map;
class Backend;
class KeyFrame;

// four tracking status
enum class FrontendStatus {INITING, TRACKING_GOOD, TRACKING_BAD, LOST};


class Frontend{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frontend> Ptr;

    Frontend();

    /* process new pair of images
     * return true if run normally
     */
    bool GrabStereoImage(const cv::Mat &leftImg, const cv::Mat &rightImg, const double &dTimeStamp);

    void SetViewer(std::shared_ptr<Viewer> viewer){
        _mpViewer = viewer;
    }

    void SetCameras(std::shared_ptr<Camera> left, std::shared_ptr<Camera> right){
        _mpCameraLeft = left;
        _mpCameraRight = right;
    }

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
    /* tracking initialization 
     * return true if success
     */
    bool StereoInit();

    /* build the initial map
     * return true if success
     */
    bool BuildInitMap();

    /* standard tracking
     * return true if success
     */
    bool Track();

    /*  get the initial value of current frame's pose using motion model
     * correspond features between last frame and current frame using LK flow
     * return the number of good tracked points
     */
    int TrackLastFrame();

    /* optimize the current frame's pose using g2o
     * the mappoints are only constrains and won't be optimized
     * return the inlier numbers 
     */
    int EstimateCurrentPose();

    /* detect new features in current image (left)
     *      when needs to create new KF (number of valid tracked features is less than a threshold)
     * return the num of features
     */
    int DetectFeatures();

    /* find the corresponding features in right image of current frame
     * return num of corresponding features found
     */
    int FindFeaturesInRight();

    /* create new keyframe (according to current frame) 
     *      and do some update work
     * return true if success
     */
    bool InsertKeyFrame();

    /* create new mappoints and insert them to the map
     * return the number of new created mappoints
     */
    int TriangulateNewPoints();

    

private:
    std::shared_ptr<ORBextractor> _mpORBextractor, _mpORBextractorInit;
    std::shared_ptr<Camera> _mpCameraLeft, _mpCameraRight;
    std::shared_ptr<Map> _mpMap;
    std::shared_ptr<Backend> _mpBackend;
    std::shared_ptr<Viewer> _mpViewer;

    FrontendStatus _mStatus = FrontendStatus::INITING;

    std::shared_ptr<Frame> _mpCurrentFrame;
    std::shared_ptr<Frame> _mpLastFrame;
    std::shared_ptr<KeyFrame> _mpReferenceKF;

    // the pose or motion variables of the current frame
    SE3 _mseRelativeMotion;
    SE3 _mseRelativeMotionToReferenceKF;

    // params for deciding the tracking status
    int _numFeaturesTrackingGood;
    int _numFeaturesTrackingBad;
    int _numFeaturesInitGood;

    bool _mbNeedUndistortion;
};




} // namespace myslam

#endif