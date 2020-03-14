#include "myslam/feature.h"

#include <opencv2/features2d.hpp>

namespace myslam{

Feature::Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp){
    mpFrame = frame;
    mkpPosition = kp;
}









}  // namespace myslam