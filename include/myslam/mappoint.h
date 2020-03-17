#ifndef MYSLAM_MAPPOINT_H
#define MYSLAM_MAPPOINT_H


#include "myslam/common_include.h"

namespace myslam{

class Frame;
class Feature;


class MapPoint{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<MapPoint> Ptr;

    MapPoint();

    MapPoint(unsigned long id, Vec3 position);

    // return the position
    Vec3 Pos() {
        std::unique_lock<std::mutex> lck(_mmutexData);
        return _mPos;
    }

    void SetPos(const Vec3 &position){
        std::unique_lock<std::mutex> lck(_mmutexData);
        _mPos = position;
    }

    // add feature in active KF which has observed this mappoint
    void AddActiveObservation(std::shared_ptr<Feature> feature);

    // return the list of the observations by active KF
    std::list<std::weak_ptr<Feature>> GetActiveObservations();

    // remove the feature from current mappoint's active observation
    void RemoveActiveObservation(std::shared_ptr<Feature> feat);


public:
    unsigned long mnId = 0;
    
    int mnObservedTimes = 0;

    bool mbIsOutlier = false;



private:
    std::mutex _mmutexData;
    
    std::list<std::weak_ptr<Feature>> _mpActiveObservations;   // 或许可以不用 list 而是 unordered_set

    Vec3 _mPos = Vec3::Zero();
};











} // namespace myslam

#endif