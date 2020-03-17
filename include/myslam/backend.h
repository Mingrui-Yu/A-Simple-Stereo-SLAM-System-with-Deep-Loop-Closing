#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"

namespace myslam{


class Backend{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    Backend();



};







} // namespace myslam

#endif