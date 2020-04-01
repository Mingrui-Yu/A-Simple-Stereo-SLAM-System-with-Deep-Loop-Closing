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
#include "myslam/loopclosing.h"
#include "myslam/ORBextractor.h"

namespace myslam{


class System{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<System> Ptr;

    System(const std::string &strConfigPath);

    void Stop();

    /* do initialization things before running
     * return true if init successfully
     */
    bool Init();

    /* process each frame
     * return true if system runs normally
     */
    bool RunStep(const cv::Mat &leftImg, const cv::Mat &rightImg, const double &dTimestamp);

    /* load the params of cameras from configuration file
     * and create Camera objects
     */ 
    void GetCamera();

    /* save the keyframe trajectory to a txt file
     */
    void SaveTrajectory(std::string &save_file);


public:
    


private:
    Frontend::Ptr _mpFrontend = nullptr;
    Backend::Ptr _mpBackend = nullptr;
    LoopClosing::Ptr _mpLoopClosing = nullptr;
    Viewer::Ptr _mpViewer = nullptr;
    std::shared_ptr<Map> _mpMap = nullptr;
    std::shared_ptr<Camera> _mpCameraLeft, _mpCameraRight;
    std::shared_ptr<ORBextractor> _mpORBextractor;

    std::string _strConfigPath;
};

}  // namespace myslam


#endif