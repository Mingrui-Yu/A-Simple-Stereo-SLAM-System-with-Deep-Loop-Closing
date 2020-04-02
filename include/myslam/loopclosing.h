#ifndef MYSLAM_LOOPCLOSING_H
#define MYSLAM_LOOPCLOSING_H

#include "myslam/common_include.h"
#include "myslam/deeplcd.h"

namespace myslam{

class KeyFrame;
class Map;
class Camera;
class ORBextractor;
class Backend;


class LoopClosing{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<LoopClosing> Ptr;

    LoopClosing();

    void Stop();

    void SetCameras(std::shared_ptr<Camera> left, std::shared_ptr<Camera> right){
        _mpCameraLeft = left;
        _mpCameraRight = right;
    }

    void SetMap(std::shared_ptr<Map> map){
        _mpMap = map;
    }

    void SetORBextractor(std::shared_ptr<ORBextractor> orb){
        _mpORBextractor = orb;
    }

    void SetBackend(std::shared_ptr<Backend> backend){
        _mpBackend = backend;
    }

    void InsertNewKeyFrame(std::shared_ptr<KeyFrame> pNewKF);


private:
    /* the main loop of loopclosing thread
     */
    void LoopClosingRun();

    /* add new KF to the KF Database
     */
    void AddToDatabase();

    /* check if there are KFs in the list which has not been processed
     */
    bool CheckNewKeyFrames();

    /* extract one KF from the list,
     * expand its features to pyramid sytle, then compute orb descriptors
     * calculate its DeepLCD descriptor vector
     */
    void ProcessNewKF();

    /* use DeepLCD to find potential KF candidate for the current KF
     * return true if successfully detect one
     */
    bool DetectLoop();

    /* match the orb descriptors
     * return true if number of valid matches is enough
    */
    bool MatchFeatures();

    /* compute the correct pose of current KF using PnP solver and g2o optimizaion
     * return true if number of inliers is enough
     */
    bool ComputeCorrectPose();

    /* use g2o to optimize the correct pose
     * this function is called in ComputeCorrectPose()
     * return the number of match inliers
     */
    int OptimizeCurrentPose();

    /* current the current pose and all previous KFs' poses according to the loop result
     */
    void LoopCorrect();

    void LoopLocalFusion();

    void PoseGraphOptimization();

    




    
private:
    std::shared_ptr<Camera> _mpCameraLeft, _mpCameraRight;
    std::shared_ptr<Map> _mpMap;
    std::shared_ptr<DeepLCD> _mpDeepLCD;
    std::shared_ptr<ORBextractor> _mpORBextractor;
    cv::Ptr<cv::DescriptorMatcher> _mpMatcher;
     std::weak_ptr<Backend> _mpBackend;

    std::thread _mthreadLoopClosing;
    std::atomic<bool>  _mbLoopClosingIsRunning;
    std::mutex _mmutexNewKFList;
    std::mutex _mmutexDatabase;

    std::shared_ptr<KeyFrame> _mpLastClosedKF = nullptr;
    std::shared_ptr<KeyFrame> _mpLastKF = nullptr;
    std::shared_ptr<KeyFrame> _mpCurrentKF = nullptr;
    std::shared_ptr<KeyFrame> _mpLoopKF = nullptr;

    std::list<std::shared_ptr<KeyFrame>> _mlNewKeyFrames;
    std::set<std::pair<int, int> > _msetValidFeatureMatches;
    Sophus::SE3d _mseCorrectedCurrentPose;
    std::map<unsigned long, std::shared_ptr<KeyFrame>> _mvDatabase;

    bool _mbNeedCorrect = false;
    bool _mbShowLoopClosingResult;

    unsigned int _mnDatabaseMinSize;
    float _similarityThres1, _similarityThres2;
    int nLevels;

};











}  // namespace myslam



















#endif