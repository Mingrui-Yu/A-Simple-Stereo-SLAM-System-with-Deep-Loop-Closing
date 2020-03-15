#include "myslam/system.h"
#include "myslam/camera.h"
#include "myslam/config.h"

namespace myslam{

// -------------------------------------------------------------------------
System::System(const std::string &strConfigPath):
        _strConfigPath(strConfigPath) {}



// -------------------------------------------------------------------------
bool System::Init(){
    // read the config file
    if (Config::SetParameterFile(_strConfigPath) == false){
        return false;
    }
    // get left and right cameras
    GetCamera();

    // create compnents and start each one's thread
    _mpFrontEnd = FrontEnd::Ptr(new FrontEnd);
    _mpViewer = Viewer::Ptr(new Viewer);

    // create links between each components
    _mpFrontEnd->SetViewer(_mpViewer);
    _mpFrontEnd->SetCameras(_cameraLeft, _cameraRight);


    return true;
}


// -------------------------------------------------------------------------
bool System::RunStep(const cv::Mat &leftImg, const cv::Mat &rightImg, const double &dTimestamp){
    
   bool success =  _mpFrontEnd->GrabStereoImage(leftImg, rightImg, dTimestamp);
    
    return success;
}


// -------------------------------------------------------------------------
void System::GetCamera(){

    float fxLeft = Config::Get<float>("Camera.right.fx");
    float  fyLeft = Config::Get<float>("Camera.right.fy");
    float cxLeft = Config::Get<float>("Camera.right.cx");
    float cyLeft = Config::Get<float>("Camera.right.cy");
    Vec3 tLeft = Vec3::Zero();

    float fxRight = Config::Get<float>("Camera.right.fx");
    float  fyRight = Config::Get<float>("Camera.right.fy");
    float cxRight = Config::Get<float>("Camera.right.cx");
    float  cyRight = Config::Get<float>("Camera.right.cy");
    float bf = Config::Get<float>("Camera.bf");
    float baseline = bf / fxRight;
    Vec3 tRight = Vec3(- baseline, 0, 0);

    _cameraLeft = Camera::Ptr(new Camera(fxLeft, fyLeft, cxLeft, cyLeft,
                                        0, SE3(SO3(), tLeft)));
    _cameraRight = Camera::Ptr(new Camera(fxRight, fyRight, cxRight, cyRight,
                                        baseline, SE3(SO3(), tRight)));
}





} // namespace myslam

