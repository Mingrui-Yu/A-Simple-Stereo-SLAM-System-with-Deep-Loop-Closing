#ifndef MYSLAM_MAP_H
#define MYSLAM_MAP_H

#include "myslam/common_include.h"


namespace myslam{

class Feature;
class MapPoint;
class KeyFrame;


class Map{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Map> Ptr;
    typedef std::unordered_map<unsigned long, std::shared_ptr<KeyFrame>> KeyFramesType;
    typedef std::unordered_map<unsigned long, std::shared_ptr<MapPoint>> MapPointsType;

    Map() {};

    // insert new keyframe to the map and the active keyframes
    void InsertKeyFrame(std::shared_ptr<KeyFrame> kf);

    // remove old keyframes from the active keyframes
    void RemoveOldActiveKeyframe();

    // insert new mappoint to the map and the active landmarks
    void InsertMapPoint(std::shared_ptr<MapPoint> map_point);

    // remove landmarks which are not observed by any active kf from active landmarks
    void RemoveOldActiveMapPoints();

    MapPointsType GetAllMapPoints();

    KeyFramesType GetAllKeyFrames();

    MapPointsType GetActiveMapPoints();

    KeyFramesType GetActiveKeyFrames();


private:

    std::mutex _mmutexData;

    std::shared_ptr<KeyFrame> _mpCurrentKF = nullptr;

    MapPointsType _mumpAllMapPoints;
    MapPointsType _mumpActiveMapPoints;

    KeyFramesType _mumpAllKeyFrames;
    KeyFramesType _mumpActiveKeyFrames;

    // settings
    unsigned int _numActiveKeyFrames = 7;
};


} // namespace myslam


#endif