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

    Map();

    /** insert new keyframe to the map and the active keyframes
     * insert new KF's mappoints to active mappoints
     * remove the old active KFs and old active mappoints
     */
    void InsertKeyFrame(std::shared_ptr<KeyFrame> kf);

    // remove mappoints which are not observed by any active kf
    void RemoveOldActiveMapPoints();

    // remove old keyframes from the active keyframes
    void RemoveOldActiveKeyframe();

    // insert new mappoint to the map and the active map
    void InsertMapPoint(std::shared_ptr<MapPoint> map_point);

     void InsertActiveMapPoint(std::shared_ptr<MapPoint> map_point);

    void RemoveMapPoint(std::shared_ptr<MapPoint> mappoint);

    /**
     * add the outlier mappoint to a list
     * the mappoints in the list will be removed from the map by RemoveAllOutlierMapPoints()
     */
   void AddOutlierMapPoint(unsigned long mpId);

   void RemoveAllOutlierMapPoints();

   
    MapPointsType GetAllMapPoints();

    KeyFramesType GetAllKeyFrames();

    MapPointsType GetActiveMapPoints();

    KeyFramesType GetActiveKeyFrames();


public:

    // avoid the conflict among different threads' operations on 
    // keyframe's poses and mappoints' positions.
    std::mutex mmutexMapUpdate;


private:

    std::mutex _mmutexData;
    std::mutex _mmutexOutlierMapPoint;

    std::shared_ptr<KeyFrame> _mpCurrentKF = nullptr;

    MapPointsType _mumpAllMapPoints;
    MapPointsType _mumpActiveMapPoints;
    std::list<unsigned long> _mlistOutlierMapPoints;

    KeyFramesType _mumpAllKeyFrames;
    KeyFramesType _mumpActiveKeyFrames;

    unsigned int _numActiveKeyFrames;
};


} // namespace myslam


#endif