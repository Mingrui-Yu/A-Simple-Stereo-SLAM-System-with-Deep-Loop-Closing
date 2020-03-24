#include "myslam/mappoint.h"
#include "myslam/feature.h"

namespace myslam{

// -------------------------------------------------------------------
MapPoint::MapPoint(){
    static unsigned long nFactoryId = 0;
    mnId = nFactoryId++;
}

// -------------------------------------------------------------------
MapPoint::MapPoint(unsigned long id, Vec3 position){
    _mPos = position;
    mnId = id;
}


// -------------------------------------------------------------------
void MapPoint::AddActiveObservation(std::shared_ptr<Feature> feature){
    std::unique_lock<std::mutex> lck(_mmutexData);
    _mpActiveObservations.push_back(feature);
    mnActiveObservedTimes++;
}

// -------------------------------------------------------------------
void MapPoint::RemoveActiveObservation(std::shared_ptr<Feature> feature){
    std::unique_lock<std::mutex> lck(_mmutexData);
    for(auto iter = _mpActiveObservations.begin(); iter != _mpActiveObservations.end(); iter++){
        if(iter->lock() == feature){
            _mpActiveObservations.erase(iter);
            // feature->mpMapPoint.reset();
            mnActiveObservedTimes--;
            break;
        }
    }
}

// -------------------------------------------------------------------
void MapPoint::RemoveObservation(std::shared_ptr<Feature> feature){
    std::unique_lock<std::mutex> lck(_mmutexData);
    for(auto iter = _mpActiveObservations.begin(); iter != _mpActiveObservations.end(); iter++){
        if(iter->lock() == feature){
            _mpActiveObservations.erase(iter);
            feature->mpMapPoint.reset();
            mnActiveObservedTimes--;
            break;
        }
    }
}


// -------------------------------------------------------------------
std::list<std::weak_ptr<Feature>> MapPoint::GetActiveObservations() {
        std::unique_lock<std::mutex> lck(_mmutexData);
        return _mpActiveObservations;
}





} // namespace myslam