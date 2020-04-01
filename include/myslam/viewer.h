#ifndef MYSLAM_VIEWER_H
#define MYSLAM_VIEWER_H

#include "myslam/common_include.h"
#include "myslam/config.h"


#include <thread>
#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>

namespace myslam {

class Frame;
class KeyFrame;
class MapPoint;
class Map;


class Viewer{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Viewer> Ptr;
    // typedef std::unordered_map<unsigned long, std::shared_ptr<KeyFrame>> KeyFramesType;
    // typedef std::unordered_map<unsigned long, std::shared_ptr<MapPoint>> MapPointsType;

    Viewer();

    void SetMap(std::shared_ptr<Map> map){
         _mpMap = map;
    }

    void Close();

    // add the current frame to viewer
    void AddCurrentFrame(std::shared_ptr<Frame> currentFrame);

    // get the information about kf/mp from the map
    // void UpdateMap();

    

private:

    void ThreadLoop();

    // show the current frame's left image and feature points
    cv::Mat PlotFrameImage();

    void FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera);

    void DrawKFsAndMPs(const bool menuShowKeyFrames, const bool menuShowPoints);

    void DrawFrame(std::shared_ptr<KeyFrame> frame, const float* color);

    void DrawFrame(std::shared_ptr<Frame> frame, const float* color);



private:
    std::thread  _mthreadViewer;
    std::mutex _mmutexViewerData;
    
    std::shared_ptr<Frame> _mpCurrentFrame = nullptr;
    std::shared_ptr<Map> _mpMap = nullptr;

    bool _mbMapUpdated = false;
    bool mbViewerRunning = true;

    const float red[3] = {1, 0, 0};
    const float green[3] = {0, 1, 0};
    const float blue[3] = {0, 0, 1};
    
    // MapPointsType _mumpAllMapPoints;
    // MapPointsType _mumpActiveMapPoints;

    // KeyFramesType _mumpAllKeyFrames;
    // KeyFramesType _mumpActiveKeyFrames;




};






}  // namespace myslam

#endif  // MYSLAM_VIEWER_H