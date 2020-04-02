#include "myslam/map.h"

#include "myslam/keyframe.h"
#include "myslam/mappoint.h"
#include "myslam/feature.h"
#include "myslam/config.h"


namespace myslam{

Map::Map(){
    _numActiveKeyFrames = Config::Get<int>("Map.activeMap.size");
}

// -------------------------------------------------------------------------------

void Map::InsertKeyFrame(std::shared_ptr<KeyFrame> kf){
    _mpCurrentKF = kf;

    {
        std::unique_lock<std::mutex> lck(_mmutexData);

        // insert keyframe
        if (_mumpAllKeyFrames.find(kf->mnKFId) == _mumpAllKeyFrames.end()){
            _mumpAllKeyFrames.insert(make_pair(kf->mnKFId, kf));
            _mumpActiveKeyFrames.insert(make_pair(kf->mnKFId, kf));
        }else{
            _mumpAllKeyFrames[kf->mnKFId] = kf;
            _mumpActiveKeyFrames[kf->mnKFId] = kf;
        }
    }

    // add the new KF to its observed mappoints' active observations
    // insert new KF's mappoints to active mappoints
    for(auto &feat: kf->mvpFeaturesLeft){
        auto mp = feat->mpMapPoint.lock();
        if(mp){
            mp->AddActiveObservation(feat);
            InsertActiveMapPoint(mp);
        }
    }

    // remove old keyframe and mappoints from the active map
    if(_mumpActiveKeyFrames.size() > _numActiveKeyFrames){
        RemoveOldActiveKeyframe();
        RemoveOldActiveMapPoints();
    }
}


// -------------------------------------------------------------------

void Map::InsertMapPoint (MapPoint::Ptr map_point) {
    std::unique_lock<std::mutex> lck(_mmutexData);

    if (_mumpAllMapPoints.find(map_point->mnId) == _mumpAllMapPoints.end()){
        _mumpAllMapPoints.insert(make_pair(map_point->mnId, map_point));
    }else {
        _mumpAllMapPoints[map_point->mnId] = map_point;
    }
}

// -------------------------------------------------------------------

void Map::InsertActiveMapPoint (MapPoint::Ptr map_point) {
    std::unique_lock<std::mutex> lck(_mmutexData);

    if (_mumpActiveMapPoints.find(map_point->mnId) == _mumpActiveMapPoints.end()){
        _mumpActiveMapPoints.insert(make_pair(map_point->mnId, map_point));
    }else {
        _mumpActiveMapPoints[map_point->mnId] = map_point;
    }
}


// -------------------------------------------------------------------

void Map::RemoveOldActiveKeyframe(){
    std::unique_lock<std::mutex> lck(_mmutexData);

    if(_mpCurrentKF == nullptr)  return;

    double maxDis = 0, minDis = 9999;
    double maxKFId = 0, minKFId = 0;

    // compute the min distance and max distance between current kf and previous active kfs
    auto Twc = _mpCurrentKF->Pose().inverse();
    for(auto &kf: _mumpActiveKeyFrames){
        if(kf.second == _mpCurrentKF) continue;
        auto dis = (kf.second->Pose() * Twc).log().norm();
        if(dis > maxDis){
            maxDis = dis;
            maxKFId = kf.first;
        } else if(dis < minDis){
            minDis = dis;
            minKFId = kf.first;
        }
    }

    // decide which kf to be removed
    const double minDisTh = 0.2;
    KeyFrame::Ptr frameToRemove = nullptr;
    if(minDis < minDisTh){
        frameToRemove = _mumpActiveKeyFrames.at(minKFId);
    } else {
        frameToRemove = _mumpActiveKeyFrames.at(maxKFId);
    }

    // LOG(INFO) << "Map: remove keyframe " << frameToRemove->mnKFId << " from the active keyframes.";

    // remove the kf and its mappoints' active observation
    _mumpActiveKeyFrames.erase(frameToRemove->mnKFId);
    for(auto &feat: frameToRemove->mvpFeaturesLeft){
        auto mp = feat->mpMapPoint.lock();
        if(mp){
            mp->RemoveActiveObservation(feat);
        }
    }

}



// -------------------------------------------------------------------

void Map::RemoveOldActiveMapPoints(){
    // if the mappoint has no active observation, then remove it from the active mappoints
    std::unique_lock<std::mutex> lck(_mmutexData);

    int cntActiveLandmarkRemoved = 0;
    for(auto iter = _mumpActiveMapPoints.begin(); iter != _mumpActiveMapPoints.end();){
        if(iter->second->mnActiveObservedTimes == 0){
            iter = _mumpActiveMapPoints.erase(iter);
            cntActiveLandmarkRemoved++;
        } else{
            ++iter;
        }
    }
    // LOG(INFO) << "Map: remove " << cntActiveLandmarkRemoved << " active landmarks";
}

// -------------------------------------------------------------------

void Map::RemoveMapPoint(std::shared_ptr<MapPoint> mappoint){

    std::unique_lock<std::mutex> lck(_mmutexData);

    unsigned long mpId = mappoint->mnId;

    // delete from all mappoints
    _mumpAllMapPoints.erase(mpId);

    // delete from active mappoints
    _mumpActiveMapPoints.erase(mpId);
}

// -------------------------------------------------------------------

void Map::AddOutlierMapPoint(unsigned long mpId){
    std::unique_lock<std::mutex> lck(_mmutexOutlierMapPoint);
    _mlistOutlierMapPoints.push_back(mpId);
}

// -------------------------------------------------------------------

void Map::RemoveAllOutlierMapPoints(){
    std::unique_lock<std::mutex> lck(_mmutexData);
    std::unique_lock<std::mutex> lck_1(_mmutexOutlierMapPoint);
    
    for(auto iter = _mlistOutlierMapPoints.begin(); iter != _mlistOutlierMapPoints.end(); iter++){
        _mumpAllMapPoints.erase(*iter);
        _mumpActiveMapPoints.erase(*iter);
    }
    _mlistOutlierMapPoints.clear();
}


// ----------------------------------------------------------------------
Map::MapPointsType Map::GetAllMapPoints(){
    std::unique_lock<std::mutex> lck(_mmutexData);
    return _mumpAllMapPoints;
}


Map::KeyFramesType Map::GetAllKeyFrames(){
    std::unique_lock<std::mutex> lck(_mmutexData);
    return _mumpAllKeyFrames;
}


Map::MapPointsType Map::GetActiveMapPoints(){
    std::unique_lock<std::mutex> lck(_mmutexData);
    return _mumpActiveMapPoints;
}


Map::KeyFramesType Map::GetActiveKeyFrames(){
    std::unique_lock<std::mutex> lck(_mmutexData);
    return _mumpActiveKeyFrames;
}

// ----------------------------------------------------------------------


} // namespace myslam