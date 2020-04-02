#include "myslam/feature.h"
#include "myslam/keyframe.h"

#include <opencv2/features2d.hpp>

namespace myslam{

Feature::Feature(std::shared_ptr<KeyFrame> kf, const cv::KeyPoint &kp){
    mpKF = kf;
    mkpPosition = kp;
}

Feature::Feature(const cv::KeyPoint &kp){
    mkpPosition = kp;
}









}  // namespace myslam