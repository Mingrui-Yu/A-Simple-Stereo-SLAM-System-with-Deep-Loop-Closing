#include "myslam/viewer.h"

#include "myslam/frame.h"
#include "myslam/feature.h"

namespace myslam{

// --------------------------------------------------------------
Viewer::Viewer(){
    _mthreadViewer = std::thread(std::bind(&Viewer::ThreadLoop, this));
}

// --------------------------------------------------------------
void Viewer::ThreadLoop(){

    std::cout << std::endl << "-------" << std::endl;
    std::cout << "Viewer Thread Loop works ..." << std::endl;

    while(1){
        {
            std::unique_lock<std::mutex> lock(_mmutexViewerData);
            if(_mpCurrentFrame){
                cv::Mat img = PlotFrameImage();
                cv::imshow("frame", img);          
            }
        }
        

        cv::waitKey(1);
        usleep(1000);
    }
    


}

// --------------------------------------------------------------
void Viewer::AddCurrentFrame(Frame::Ptr currentFrame){
    std::unique_lock<std::mutex> lock(_mmutexViewerData);
    _mpCurrentFrame = currentFrame;
}

// --------------------------------------------------------------
cv::Mat Viewer::PlotFrameImage(){
    cv::Mat img_out;
    cv::cvtColor(_mpCurrentFrame->mLeftImg, img_out, CV_GRAY2BGR);
    for (size_t i = 0, N = _mpCurrentFrame->mvpFeaturesLeft.size(); i < N; ++i){
            auto feat = _mpCurrentFrame->mvpFeaturesLeft[i];
            cv::circle(img_out, feat->mkpPosition.pt, 2, cv::Scalar(0,250,0), 2);
    }
    return img_out;
}

}  // namespace myslam
