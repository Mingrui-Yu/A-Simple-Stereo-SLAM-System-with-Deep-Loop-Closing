#ifndef MYSLAM_SYSTEM_H
#define MYSLAM_SYSTEM_H

#include "myslam/common_include.h"
#include "myslam/config.h"
#include "myslam/frame.h"
#include "myslam/frontend.h"
#include "myslam/viewer.h"


namespace myslam{

/* 
 * system 对外接口
 */

class Camera;


class System{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<System> Ptr;

    System(const std::string &strConfigPath);

    // do initialization things before run
    bool Init();

    bool RunStep(const cv::Mat &leftImg, const cv::Mat &rightImg, const double &dTimestamp);

    void GetCamera();

public:
    



private:
    std::string _strConfigPath;

    FrontEnd::Ptr _mpFrontEnd = nullptr;
    Viewer::Ptr _mpViewer = nullptr;

    std::shared_ptr<Camera> _cameraLeft, _cameraRight;

};

}  // namespace myslam


#endif