#ifndef MYSLAM_FEATURE_H
#define MYSLAM_FEATURE_H


#include "myslam/common_include.h"

namespace myslam{

// forward declaration
class Frame;
class MapPoint;
class KeyFrame;

class Feature{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Feature> Ptr;

    Feature() {}

    Feature(const cv::KeyPoint &kp);
    // Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp);


public:
    // std::weak_ptr<Frame> mpFrame;
    std::weak_ptr<KeyFrame> mpKF;
    cv::KeyPoint mkpPosition; 
    std::vector<cv::KeyPoint> mvPyramidKeyPoints;

    std::weak_ptr<MapPoint> mpMapPoint; 

    bool mbIsOnLeftFrame = true; // true: on left frame; false: on right frame;
    bool mbIsOutlier = false;  


};



}  // namespace myslam

#endif