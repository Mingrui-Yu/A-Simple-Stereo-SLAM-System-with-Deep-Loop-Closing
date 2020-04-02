#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

#include "myslam/common_include.h"
#include <algorithm> 

namespace myslam{

/**
 * linear triangulation with SVD
 * @param poses     poses,
 * @param points    points in normalized plane
 * @param pt_world  triangulated point in the world
 * @return true if success
 */
inline bool triangulation(const std::vector<SE3> &poses,
                   const std::vector<Vec3> points, Vec3 &pt_world) {
    MatXX A(2 * poses.size(), 4);
    VecX b(2 * poses.size());
    b.setZero();
    for (size_t i = 0; i < poses.size(); ++i) {
        Mat34 m = poses[i].matrix3x4();
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
    }
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
        return true;
    } // give up the bad solution
    return false;
}

inline Vec2 toVec2(const cv::Point2f p){
    return Vec2(p.x, p.y);
}

inline Vec3 toVec3(const cv::Point3f p){
    return Vec3(p.x, p.y, p.z);
}

// ----------------------------------------------------------------------------------------------

// unused 
inline bool Kmeans(std::vector<float> &data){
    if(data.size() < 2) return false;

    float meanLow = *(std::min_element(data.begin(), data.end()));
    float meanHigh = *(std::max_element(data.begin(), data.end()));
    bool bChanged = true;
    int cntLow = 0, cntHigh = 0;
    std::vector<float> type(data.size());
    while(bChanged){
        bChanged = false;
        for(size_t i = 0, N = data.size(); i < N; i++){
            float disToLow = std::abs(data[i] - meanLow);
            float disToHigh = std::abs(data[i] - meanHigh);
            if(disToLow < disToHigh){
                if(type[i] == 1) 
                    bChanged = true;
                type[i] = 0;
            }
            else{
                if(type[i] == 0) 
                    bChanged = true;
                type[i] = 1;
            }
        }      

        meanLow = 0;
        meanHigh = 0;  
         cntLow = 0;
         cntHigh = 0;
        for(size_t i = 0, N = data.size(); i < N; i++){
            if(type[i] == 0){
                meanLow += data[i];
                cntLow++;
            }else{
                meanHigh += data[i];
                cntHigh++;
            }
        }      
        meanLow /= cntLow;
        meanHigh /= cntHigh;
    }

    for(size_t i = 0, N = data.size(); i < N; i++){
        std::cout << data[i] << " ";
    }      
    std::cout << std::endl;
    for(size_t i = 0, N = data.size(); i < N; i++){
        std::cout << type[i] << " ";
    }      
    std::cout << std::endl;

    if(cntHigh > 3){
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------

// compute the mean and standard variance of a vector
inline std::pair<double, double> VectorMeanAndVariance(const std::vector<double> v){
    double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
    double m =  sum / v.size();
    double accum = 0.0;
    std::for_each (std::begin(v), std::end(v), [&](const double d) {
        accum += (d - m) * (d - m);
    });
    double stdev = sqrt(accum / (v.size()-1));

    std::pair<double, double> pairMeanAndVariance;
    pairMeanAndVariance.first = m;
    pairMeanAndVariance.second = stdev;

    return pairMeanAndVariance;
}

inline std::pair<float, float> VectorMeanAndVariance(const std::vector<float> v){
    float sum = std::accumulate(std::begin(v), std::end(v), 0.0);
    float m =  sum / v.size();
    float accum = 0.0;
    std::for_each (std::begin(v), std::end(v), [&](const float d) {
        accum += (d - m) * (d - m);
    });
    float stdev = sqrt(accum / (v.size()-1));

    std::pair<float, float> pairMeanAndVariance;
    pairMeanAndVariance.first = m;
    pairMeanAndVariance.second = stdev;

    return pairMeanAndVariance;
}



} // namespace myslam




# endif