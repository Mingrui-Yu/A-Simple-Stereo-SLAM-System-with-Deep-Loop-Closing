#include "myslam/map.h"
#include "myslam/keyframe.h"

namespace myslam{

void Map::InsertKeyFrame(std::shared_ptr<KeyFrame> kf){
    _mpCurrentKF = kf;

    if (_mumpAllKeyFrames.find(kf->mnKFId) == _mumpAllKeyFrames.end()){
        _mumpAllKeyFrames.insert(make_pair(kf->mnKFId, kf));
        _mumpActiveKeyFrames.insert(make_pair(kf->mnKFId, kf));
    }else{
        _mumpAllKeyFrames[kf->mnKFId] = kf;
        _mumpActiveKeyFrames[kf->mnKFId] = kf;
    }
}

if(_mumpActiveKeyFrames.size() > _numActiveKeyFrames){
    // tomorrow's work
}





} // namespace myslam