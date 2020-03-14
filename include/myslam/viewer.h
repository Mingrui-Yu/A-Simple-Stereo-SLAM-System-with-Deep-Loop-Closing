#ifndef MYSLAM_VIEWER_H
#define MYSLAM_VIEWER_H

#include "myslam/common_include.h"
#include "myslam/config.h"
#include "myslam/frame.h"

#include <thread>
#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

namespace myslam {


class Viewer{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Viewer> Ptr;

    Viewer();

    void ThreadLoop();

    cv::Mat PlotFrameImage();

    void AddCurrentFrame(Frame::Ptr currentFrame);



private:
    std::thread  _mthreadViewer;
    std::mutex _mmutexViewerData;
    
    Frame::Ptr _mpCurrentFrame = nullptr;

    const float red[3] = {1, 0, 0};
    const float green[3] = {0, 1, 0};
    const float blue[3] = {0, 0, 1};
    
    




};






}  // namespace myslam

#endif  // MYSLAM_VIEWER_H