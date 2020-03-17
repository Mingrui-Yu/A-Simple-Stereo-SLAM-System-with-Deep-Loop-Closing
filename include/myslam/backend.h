#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"
#include "myslam/map.h"

namespace myslam{

class Camera;
class Viewer;
class Map;

class Backend{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    Backend();

    void UpdateMap();

    void Stop();

    void SetViewer(std::shared_ptr<Viewer> viewer){
        _mpViewer = viewer;
    }

    // set left and right camera
    void SetCameras(std::shared_ptr<Camera> left, std::shared_ptr<Camera> right){
        _mpCameraLeft = left;
        _mpCameraRight = right;
    }

    // set the map
    void SetMap(std::shared_ptr<Map> map){
        _mpMap = map;
    }

private:
    void BackendLoop();

    void OptimizeActiveMap(Map::KeyFramesType &kfs, Map::MapPointsType &mps);

private:
    std::thread _mthreadBackend;
    std::atomic<bool>  _mbBackendIsRunning;
    std::condition_variable _mapUpdate;
    std::mutex _mmutexData;

    std::shared_ptr<Camera> _mpCameraLeft, _mpCameraRight;

    std::shared_ptr<Map> _mpMap;

    // Other thread Pointers
    std::shared_ptr<Viewer> _mpViewer;

    


};







} // namespace myslam

#endif