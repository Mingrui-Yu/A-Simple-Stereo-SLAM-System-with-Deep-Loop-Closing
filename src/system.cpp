#include "myslam/system.h"

namespace myslam{

System::System(const std::string &strConfigPath):
        _strConfigPath(strConfigPath) {}


bool System::Init(){
    // read the config file
    if (Config::SetParameterFile(_strConfigPath) == false){
        return false;
    }

    // create compnents and start each one's thread
    _mpFrontEnd = FrontEnd::Ptr(new FrontEnd);
    _mpViewer = Viewer::Ptr(new Viewer);

    // create links between each components
    _mpFrontEnd->SetViewer(_mpViewer);


    return true;
}

bool System::RunStep(const cv::Mat &leftImg, const cv::Mat &rightImg, const double &dTimestamp){
    
   bool success =  _mpFrontEnd->GrabStereoImage(leftImg, rightImg, dTimestamp);
    
    return success;
}






} // namespace myslam

