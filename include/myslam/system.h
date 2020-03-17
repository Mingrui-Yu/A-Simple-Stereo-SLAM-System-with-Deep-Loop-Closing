#ifndef MYSLAM_SYSTEM_H
#define MYSLAM_SYSTEM_H

#include "myslam/common_include.h"
#include "myslam/config.h"
#include "myslam/frame.h"
#include "myslam/frontend.h"
#include "myslam/backend.h"
#include "myslam/viewer.h"
#include "myslam/map.h"
#include "myslam/camera.h"

namespace myslam{

/* 
 * system 对外接口
 */


class System{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<System> Ptr;

    System(const std::string &strConfigPath);

    void Stop();

    // do initialization things before run
    bool Init();

    bool RunStep(const cv::Mat &leftImg, const cv::Mat &rightImg, const double &dTimestamp);

    void GetCamera();

public:
    



private:
    std::string _strConfigPath;

    Frontend::Ptr _mpFrontend = nullptr;
    Backend::Ptr _mpBackend = nullptr;
    Viewer::Ptr _mpViewer = nullptr;
    std::shared_ptr<Map> _mpMap = nullptr;
    std::shared_ptr<Camera> _mpCameraLeft, _mpCameraRight;

};

}  // namespace myslam


#endif