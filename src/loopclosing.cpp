#include "myslam/loopclosing.h"

#include "myslam/feature.h"
#include "myslam/mappoint.h"
#include "myslam/keyframe.h"
#include "myslam/map.h"
#include "myslam/tools.h"
#include "myslam/config.h"
#include "myslam/algorithm.h"
#include "myslam/ORBextractor.h"
#include "myslam/camera.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>


namespace myslam{

// -----------------------------------------------------------------------------------
LoopClosing::LoopClosing(){
    _mbLoopClosingIsRunning.store(true);
    _mpDeepLCD = DeepLCD::Ptr(new DeepLCD);

    _mthreadLoopClosing = std::thread(std::bind(&LoopClosing::LoopClosingRun, this));

    _similarityThres1 = Config::Get<float>("LCD.similarityScoreThreshold.high");
    _similarityThres2 = Config::Get<float>("LCD.similarityScoreThreshold.low");

    int numORBNewFeatures = Config::Get<int>("ORBextractor.nNewFeatures");
    float fScaleFactor = Config::Get<float>("ORBextractor.scaleFactor");
    int nLevels = Config::Get<int>("ORBextractor.nLevels");

    _mpORBdescriptor = cv::ORB::create(numORBNewFeatures * 2, fScaleFactor, nLevels); // numORBNewFeatures, fScaleFactor, nLevels
    _mpMatcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
}

// -----------------------------------------------------------------------------------

void LoopClosing::Stop(){
    while(CheckNewKeyFrames()){
        usleep(1000);
    }
    _mbLoopClosingIsRunning.store(false);
    _mthreadLoopClosing.join();
}


// -------------------------------------------------------------------------------------
void LoopClosing::LoopClosingRun(){
    while(_mbLoopClosingIsRunning.load()){
        if(CheckNewKeyFrames()){
            
            // extract one KF to process from the Database
            ProcessNewKF();
                
            // try to find the loop KF for the current KF
            if(_mvDatabase.size() > _mnDatabaseMinSize){
                if(DetectLoop()){
                    if(MatchFeatures()){
                        if(ComputeSE3()){
                            LoopFusion();
                        }
                    }
                }
            }

            if(1){
                AddToDatabase();
            }
        }
    }
}

// -------------------------------------------------------------------------------------
void LoopClosing::ProcessNewKF(){
    std::unique_lock<std::mutex> lck(_mmutexNewKFList);
    _mpCurrentKF = _mlNewKeyFrames.front();
    _mlNewKeyFrames.pop_front();

    // calculate the whole image's descriptor vector with deeplcd
    _mpCurrentKF->mpDescrVector = _mpDeepLCD->calcDescrOriginalImg(_mpCurrentKF->mImageLeft);
    
    // calculate the orb descriptors of the keypoints using function in "myslam/ORBextractor.h"
    _mpORBextractor->CalcDescriptors(_mpCurrentKF->mImageLeft, 
            _mpCurrentKF->GetKeyPoints(), _mpCurrentKF->mORBDescriptors);
    
    // _mpCurrentKF->mImageLeft.release();  // if neeeded
}

// -------------------------------------------------------------------------------------
bool LoopClosing::DetectLoop(){
    // std::unique_lock<std::mutex> lck(_mmutexDatabase);
    std::vector<float> vScores;
    float maxScore = 0;
    int cntSuspected = 0;
    unsigned long bestId = 0;
    for(auto &db:  _mvDatabase){
        if(_mpCurrentKF->mnKFId - db.first < 20) break;
        float similarityScore = _mpDeepLCD->score(_mpCurrentKF->mpDescrVector, db.second->mpDescrVector);
        if(similarityScore > maxScore){
            maxScore = similarityScore;
            bestId = db.first;
        }
        if(similarityScore > _similarityThres2){
            cntSuspected++;
        }
    }

    if(maxScore > _similarityThres1 && cntSuspected <= 3){
        _mpLoopKF = _mvDatabase.at(bestId);

        LOG(INFO) << "find potential Candidate KF: current KF " 
            << _mpCurrentKF->mnFrameId << ", candiate KF " << _mpLoopKF->mnFrameId << ", "
            << _mpLoopKF->mnKFId
            << ", score: " << maxScore;

        return true;
    }
    return false;
}



// -------------------------------------------------------------------------------------

bool LoopClosing::MatchFeatures(){
    std::vector<cv::DMatch> matches;
    _mvGoodFeatureMatches.clear();
    _mpMatcher->match(_mpCurrentKF->mORBDescriptors, _mpLoopKF->mORBDescriptors, matches);
    
    auto min_max = std::minmax_element(matches.begin(), matches.end(),
                [] (const cv::DMatch &m1, const cv::DMatch &m2) {return m1.distance < m2.distance;});
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    LOG(INFO) << "--Max dist = " << max_dist;
    LOG(INFO) << "--Min dist = " << min_dist;

    for (int i=0; i<_mpCurrentKF->mORBDescriptors.rows; i++){
        if(matches[i].distance <= max(2*min_dist, 30.0)){
            _mvGoodFeatureMatches.push_back(matches[i]);
        }
    }

    if(_mvGoodFeatureMatches.size() < 30){
        return false;
    }


    LOG(INFO) << "match number: " << matches.size() ;
    LOG(INFO) << "good match number: " << _mvGoodFeatureMatches.size();
    cv::Mat img_goodmatch;
    cv::drawMatches(_mpCurrentKF->mImageLeft, _mpCurrentKF->GetKeyPoints(), 
            _mpLoopKF->mImageLeft, _mpLoopKF->GetKeyPoints(), 
            _mvGoodFeatureMatches, img_goodmatch);
    cv::resize(img_goodmatch, img_goodmatch, Size(), 0.5, 0.5);
    cv::imshow("good matches", img_goodmatch);
    cv::waitKey(1);
    
    return true;
}


// ----------------------------------------------------------------------------------

bool LoopClosing::ComputeSE3(){

    // prepare the data for PnP solver
    std::vector<cv::Point3f> vLoopPoints3d;
    std::vector<cv::Point2f> vCurrentPoints2d;
    std::vector<cv::Point2f> vLoopPoints2d;

    for(cv::DMatch &match: _mvGoodFeatureMatches){
        auto mp = _mpLoopKF->mvpFeaturesLeft[match.trainIdx]->mpMapPoint.lock();
        if(mp){
            vCurrentPoints2d.push_back(
                _mpCurrentKF->mvpFeaturesLeft[match.queryIdx]->mkpPosition.pt);
            Vec3 pos = mp->Pos();
            // LOG(INFO) << "mappoint pos: " << pos;
            vLoopPoints3d.push_back(cv::Point3f(pos(0), pos(1), pos(2)));
            vLoopPoints2d.push_back(_mpLoopKF->mvpFeaturesLeft[match.trainIdx]->mkpPosition.pt);
        }
    }

    LOG(INFO) << "number of valid 3d-2d pairs: " << vLoopPoints3d.size();

    if(vLoopPoints3d.size() < 20) 
        return false;

    cv::Mat rvec, tvec, R, K;
    cv::eigen2cv(_mpCameraLeft->K(), K);

    Eigen::Matrix3d Reigen;
    Eigen::Vector3d teigen;

    // cv::Mat essentialMatrix, mask;
    // essentialMatrix = cv::findEssentialMat(vLoopPoints2d, vCurrentPoints2d, K, RANSAC, 0.95, 5.991, mask);
    // cv::recoverPose(essentialMatrix, vLoopPoints2d, vCurrentPoints2d, K, R, tvec, mask);

    // cv::cv2eigen(R, Reigen);
    // cv::cv2eigen(tvec, teigen);
    // Sophus::SE3d relativePose(Reigen, teigen);
    // currentPose = relativePose * _mpLoopKF->Pose();
    // LOG(INFO) << "loop KF pose matrix: \n" << _mpLoopKF->Pose().matrix();
    // LOG(INFO) << "current KF pose: by essential matrix: \n" << currentPose.matrix();

    cv::Mat inliers;
    bool success = cv::solvePnPRansac(vLoopPoints3d, vCurrentPoints2d, 
            K, cv::Mat(), rvec, tvec, false, 100, 5.991, 0.95, inliers); // SOLVEPNP_EPNP
    if(success && inliers.rows > 20){
        cv::Rodrigues(rvec, R);
        cv::cv2eigen(R, Reigen);
        cv::cv2eigen(tvec, teigen);
        _mseCorrectedCurrentPose = Sophus::SE3d(Reigen, teigen);
        LOG(INFO) << "by pnp: \n" << _mseCorrectedCurrentPose.matrix();

         // show the reprojection result
        // std::vector<cv::Point2f> vReprojectionPoints2d;
        // cv::projectPoints(vLoopPoints3d, rvec, tvec, K, cv::Mat(), vReprojectionPoints2d);
        // LOG(INFO) << "current points size: " << vCurrentPoints2d.size();
        // LOG(INFO) << "reprojection points size: " << vReprojectionPoints2d.size();
        // LOG(INFO) << "inliers size: " << inliers.rows;
        // cv::Mat imgOut;
        // cv::cvtColor(_mpCurrentKF->mImageLeft, imgOut,cv::COLOR_GRAY2RGB);
        // for(int i = 0; i < inliers.rows; i++){
        //     int index = inliers.at<int>(i, 0);
        //     cv::circle(imgOut, vCurrentPoints2d[index], 5, cv::Scalar(0, 0, 255), -1);
        //     cv::line(imgOut, vCurrentPoints2d[index], vReprojectionPoints2d[index], cv::Scalar(255, 0, 0), 2);
        // }
        // cv::imshow("output", imgOut);
        // cv::waitKey(1);

        return true;
    }


    return false;
}

// -------------------------------------------------------------------------------------

void LoopClosing::LoopFusion(){

    for(auto &feat: _mpCurrentKF->mvpFeaturesLeft){
        auto mp = feat->mpMapPoint.lock();
        if(mp){
            Vec3 posCamera = _mpCurrentKF->Pose() * mp->Pos();
            mp->SetPos(_mseCorrectedCurrentPose.inverse() * posCamera);
        }
    }
    _mpCurrentKF->SetPose(_mseCorrectedCurrentPose);
}





// -------------------------------------------------------------------------------------

void LoopClosing::AddToDatabase(){
    // std::unique_lock<std::mutex> lck(_mmutexDatabase);
    // LOG(INFO) << "add KF " <<  _mpCurrentKF->mnKFId << " to database.";
    
    // add curernt KF to the Database
    _mvDatabase.insert({_mpCurrentKF->mnKFId, _mpCurrentKF});
    _mpLastKF = _mpCurrentKF;
}

// ----------------------------------------------------------------------------------


// -----------------------------------------------------------------------------------

bool LoopClosing::CheckNewKeyFrames(){
    std::unique_lock<std::mutex> lck(_mmutexNewKFList);
    return(!_mlNewKeyFrames.empty());
}

// ---------------------------------------------------------------------------------------

void LoopClosing::InsertNewKeyFrame(KeyFrame::Ptr pNewKF){
    std::unique_lock<std::mutex> lck(_mmutexNewKFList);
    // the requirement of inserting new KF is:
    // 1. more than 10 KFs after the last successful loop closing
    if(_mpLastClosedKF == nullptr 
        || pNewKF->mnKFId - _mpLastClosedKF->mnKFId > 10){
            _mlNewKeyFrames.push_back(pNewKF);
        }
}

// ---------------------------------------------------------------------------------------








}   // namespace myslam