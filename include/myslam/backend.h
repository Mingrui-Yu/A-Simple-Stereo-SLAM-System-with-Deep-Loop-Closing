#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"
#include "myslam/map.h"


namespace myslam{

class Camera;
class Viewer;
class Map;
class Frame;
class KeyFrame;
class LoopClosing;

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

    void SetLoopClosing(std::shared_ptr<LoopClosing> lp){
        _mpLoopClosing = lp;
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

    void InsertKeyFrame(std::shared_ptr<KeyFrame> pKF);

    void RequestPause();

    bool IfHasPaused();

    void Resume();

private:
    bool CheckNewKeyFrames();

    void ProcessNewKeyFrame();

    void BackendLoop();

    void OptimizeActiveMap();

private:
    bool _mbNeedOptimization = false;

    std::thread _mthreadBackend;
    std::atomic<bool>  _mbBackendIsRunning;
    std::atomic<bool>  _mbRequestPause;
    std::atomic<bool>  _mbFinishedOneLoop;
    std::condition_variable _mapUpdate;
    std::mutex _mmutexData;
    std::mutex _mmutexNewKF;
    std::mutex _mmutexStop;

    std::shared_ptr<Camera> _mpCameraLeft, _mpCameraRight;

    std::shared_ptr<Map> _mpMap;

     

    // Other thread Pointers
    std::shared_ptr<Viewer> _mpViewer = nullptr;
    std::shared_ptr<LoopClosing> _mpLoopClosing = nullptr;

    std::list<std::shared_ptr<KeyFrame>> _mlNewKeyFrames;

    std::shared_ptr<KeyFrame> _mpLastKF = nullptr;
    std::shared_ptr<KeyFrame> _mpCurrentKF;
    std::shared_ptr<Frame> _mpCurrentFrame;
    


};







} // namespace myslam

#endif