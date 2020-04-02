#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"


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

    // void UpdateMap();

    /* ask the backend thread to stop
    */
    void Stop();

    void SetViewer(std::shared_ptr<Viewer> viewer){
        _mpViewer = viewer;
    }

    void SetLoopClosing(std::shared_ptr<LoopClosing> lp){
        _mpLoopClosing = lp;
    }

    void SetCameras(std::shared_ptr<Camera> left, std::shared_ptr<Camera> right){
        _mpCameraLeft = left;
        _mpCameraRight = right;
    }

    void SetMap(std::shared_ptr<Map> map){
        _mpMap = map;
    }

    /* insert KeyFrame from frontend
     * add inserted KeyFrame to a list, and it will be processed later
    */
    void InsertKeyFrame(std::shared_ptr<KeyFrame> pKF);

    /* ask the backend thread to pause
     */
    void RequestPause();

    /* return true if the backend thread has paused
     */
    bool IfHasPaused();

    /* ask the backend thread to resume running
    */
    void Resume();


private:
    /* check if there are KFs in the list which has not been processed
     */
    bool CheckNewKeyFrames();

    /* extract one new KF from the list and process it
     * insert it into the map
     * insert it into the loopclosing thread
     */
    void ProcessNewKeyFrame();

    /* the main loop of backend thread
     */
    void BackendRun();

    /* g2o optimization
     *  optmize the KFs and mappoints in active map
     */
    void OptimizeActiveMap();


private:

    std::shared_ptr<Camera> _mpCameraLeft, _mpCameraRight;
    std::shared_ptr<Map> _mpMap;
    std::shared_ptr<Viewer> _mpViewer = nullptr;
    std::shared_ptr<LoopClosing> _mpLoopClosing = nullptr;

    std::thread _mthreadBackend;
    std::atomic<bool>  _mbBackendIsRunning;
    std::atomic<bool>  _mbRequestPause;
    std::atomic<bool>  _mbHasPaused;
    // std::condition_variable _mapUpdate;
    // std::mutex _mmutexData;
    std::mutex _mmutexNewKF;
    std::mutex _mmutexStop;
    bool _mbNeedOptimization = false;

    std::list<std::shared_ptr<KeyFrame>> _mlNewKeyFrames;

    std::shared_ptr<KeyFrame> _mpCurrentKF = nullptr;
    

};


} // namespace myslam

#endif