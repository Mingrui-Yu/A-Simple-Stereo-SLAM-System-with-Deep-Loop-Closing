#ifndef MYSLAM_CONFIG_H
#define MYSLAM_CONFIG_H

#include "myslam/common_include.h"

namespace myslam{

class Config {
public:

    ~Config();

    // set a new config file
    static bool SetParameterFile(const std::string &filename);

    // access the parameter values
    template <typename T>
    static T Get(const std::string &key){
        return T(Config::config_->file_[key]);
    }
    


private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;

    Config() {}  // private constructor makes a singleton
};

}

#endif