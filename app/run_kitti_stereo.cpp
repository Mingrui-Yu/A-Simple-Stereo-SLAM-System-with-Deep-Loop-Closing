#include <myslam/system.h>
#include <myslam/common_include.h>

#include <iostream>
#include <fstream>
#include<iomanip>
#include <chrono>

#include<opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>



// DEFINE_string(config_file, "./config/default.yaml", "config file path");

using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);


// ----------------------------------------------------------------------------------------------------------------------------------------------------

int main(int argc, char **argv){
    // 解析命令行参数
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if(argc != 3){
        std::cerr << endl << "Usage: ./bin/run_kitti_stereo  path_to_config  path_to_sequence" << std::endl;
        return 1;
    }

    std::string strConfigPath = argv[1];
    std::string strSequencePath = argv[2];

    // load sequence frames
    std::vector<std::string> vstrImageLeft, vstrImageRight;
    std::vector<double> vTimestamps;
    LoadImages(strSequencePath, vstrImageLeft, vstrImageRight, vTimestamps);
    const int nImages = vstrImageLeft.size();

    // create the slam system
    myslam::System::Ptr slam(new myslam::System(strConfigPath));
    if(!slam->Init()){
        std::cerr << "cannot initialize the system!" << std::endl;
    };
    
    // start processing sequence
    std::cout << std::endl << "-------" << std::endl;
    std::cout << "Start processing sequence ..." << std::endl;
    std::cout << "Total number of images in the sequence: " << nImages << std::endl;
    cv::Mat imgLeft, imgRight;
    double dTimeStamp;

    std::chrono::steady_clock::time_point t_start, t_end, t1, t2;
    std::chrono::duration<double> time_used_total, time_used;

    t_start = std::chrono::steady_clock::now();
    for(int ni=0; ni < nImages; ni++){
        t1 = std::chrono::steady_clock::now();
        if(ni % 100 == 99) std::cout << "ni = " << ni + 1 << std::endl;
        
        imgLeft = cv::imread(vstrImageLeft[ni], cv::IMREAD_GRAYSCALE);
        imgRight = cv::imread(vstrImageRight[ni], cv::IMREAD_GRAYSCALE);
        dTimeStamp = vTimestamps[ni];

        if(imgLeft.empty()){
            std::cerr << std::endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << std::endl;
            return 1;
        }

        slam->RunStep(imgLeft, imgRight, dTimeStamp);
        t2 = std::chrono::steady_clock::now();
        time_used = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
        
        LOG(INFO) << "time cost for frame " << ni <<": " << time_used.count()  << "s";
        // if(ni==3) break;
    }
    t_end = std::chrono::steady_clock::now();
    time_used_total = std::chrono::duration_cast <std::chrono::duration<double>> (t_end - t_start);
    
    slam->Stop();
    
    std::cout << std::endl << "-------" << std::endl;
    std::cout << "system stop." << std::endl;
    std::cout << "total time cost: " << time_used_total.count() 
        << ", average fps: " << nImages / time_used_total.count()  << std::endl;
    return 0;
}




// ----------------------------------------------------------------------------------------------------------------------------------------------------

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps){
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof()){
        string s;
        getline(fTimes,s);
        if(!s.empty()){
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++){
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}
