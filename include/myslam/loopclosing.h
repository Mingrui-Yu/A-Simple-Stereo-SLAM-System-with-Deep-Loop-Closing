#ifndef MYSLAM_LOOPCLOSING_H
#define MYSLAM_LOOPCLOSING_H

#include "myslam/common_include.h"
#include "myslam/deeplcd.h"

namespace myslam{

class KeyFrame;
class Map;
class Camera;
class ORBextractor;


class LoopClosing{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<LoopClosing> Ptr;

    LoopClosing();

    void Stop();

    // set left and right camera
    void SetCameras(std::shared_ptr<Camera> left, std::shared_ptr<Camera> right){
        _mpCameraLeft = left;
        _mpCameraRight = right;
    }

    // set the map
    void SetMap(std::shared_ptr<Map> map){
        _mpMap = map;
    }

    void SetORBextractor(std::shared_ptr<ORBextractor> orb){
        _mpORBextractor = orb;
    }

    void InsertNewKeyFrame(std::shared_ptr<KeyFrame> pNewKF);

private:
    void LoopClosingRun();

    void AddToDatabase();

    bool CheckNewKeyFrames();

    void ProcessNewKF();

    bool DetectLoop();

    bool MatchFeatures();

    bool ComputeSE3();


    
private:
    std::thread _mthreadLoopClosing;
    std::atomic<bool>  _mbLoopClosingIsRunning;
    std::mutex _mmutexNewKFList;
    std::mutex _mmutexDatabase;

    std::shared_ptr<Camera> _mpCameraLeft, _mpCameraRight;
    std::shared_ptr<Map> _mpMap;
    std::shared_ptr<DeepLCD> _mpDeepLCD;
    std::shared_ptr<ORBextractor> _mpORBextractor;
    cv::Ptr<cv::DescriptorExtractor> _mpORBdescriptor;
     cv::Ptr<cv::DescriptorMatcher> _mpMatcher;

    std::shared_ptr<KeyFrame> _mpLastClosedKF = nullptr;
    std::shared_ptr<KeyFrame> _mpLastKF = nullptr;
    std::shared_ptr<KeyFrame> _mpCurrentKF = nullptr;
    std::shared_ptr<KeyFrame> _mpLoopKF = nullptr;
    std::vector<cv::DMatch> _mvGoodFeatureMatches;
    // DeepLCD::DescrVector _mCurrentDescrVector;
    std::list<std::shared_ptr<KeyFrame>> _mlNewKeyFrames;

    std::map<unsigned long, std::shared_ptr<KeyFrame>> _mvDatabase;

    unsigned int _mnDatabaseMinSize = 50;
    float _similarityThres1, _similarityThres2;

};











}  // namespace myslam



















#endif