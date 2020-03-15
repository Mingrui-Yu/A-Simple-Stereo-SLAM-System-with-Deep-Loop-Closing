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

    MapPoint() {}

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

    // add feature which has observed this mappoint
    void AddObservation(std::shared_ptr<Feature> feature);

    // remove feature  from the observing features
    void RemoveObservation(std::shared_ptr<Feature> feature);

    // return the list of the observations
    std::list<std::weak_ptr<Feature>> GetObservations();


public:
    unsigned long mnId = 0;
    std::list<std::weak_ptr<Feature>> _mpObservations;   // 或许可以不用 list 而是 unordered_set
    int mnObservedTimes = 0;



private:
    std::mutex _mmutexData;

    Vec3 _mPos = Vec3::Zero();
};











} // namespace myslam

#endif