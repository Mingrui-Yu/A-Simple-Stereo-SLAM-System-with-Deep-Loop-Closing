#ifndef MYSLAM_MAP_H
#define MYSLAM_MAP_H

#include "myslam/common_include.h"


namespace myslam{

class MapPoint;
class KeyFrame;

class Map{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Map> Ptr;
    typedef std::unordered_map<unsigned long, std::shared_ptr<KeyFrame>> KeyFramesType;
    typedef std::unordered_map<unsigned long, std::shared_ptr<MapPoint>> LandmarksType;

    Map() {};

    void InsertKeyFrame(std::shared_ptr<KeyFrame> kf);

private:

    std::shared_ptr<KeyFrame> _mpCurrentKF = nullptr;

    LandmarksType _mumpAllLandmarks;
    LandmarksType _mumpActiveLandmarks;
    KeyFramesType _mumpAllKeyFrames;
    KeyFramesType _mumpActiveKeyFrames;


    // settings
    unsigned int _numActiveKeyFrames = 7;
};


} // namespace myslam


#endif