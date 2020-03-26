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
#include "myslam/backend.h"

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
            bool bConfirmedLoopKF = false;
            if(_mvDatabase.size() > _mnDatabaseMinSize){
                if(DetectLoop()){
                    if(MatchFeatures()){
                        bConfirmedLoopKF = ComputeSE3();
                        if(bConfirmedLoopKF){
                            LoopFusion();
                        }
                    }
                }
            }

            if(! bConfirmedLoopKF){
                AddToDatabase();
            }
        }
        usleep(1000);
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

    std::vector<float> vScores;
    float maxScore = 0;
    int cntSuspected = 0;
    unsigned long bestId = 0;

    for(auto &db:  _mvDatabase){

        // avoid comparing with recent KFs
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

    // require high similarity score
    // however, if there are too many high similarity scores, it means that current KF is not specific, then skip it
    if(maxScore < _similarityThres1 || cntSuspected > 3){
        return false;
    }

    _mpLoopKF = _mvDatabase.at(bestId);

    LOG(INFO) << "find potential Candidate KF: current KF " 
        << _mpCurrentKF->mnFrameId << ", candiate KF " << _mpLoopKF->mnFrameId << ", "
        << _mpLoopKF->mnKFId
        << ", score: " << maxScore;

    return true;
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
    _mmatMatchInliers.release();

    bool success = cv::solvePnPRansac(vLoopPoints3d, vCurrentPoints2d, 
            K, cv::Mat(), rvec, tvec, false, 200, 5.991, 0.99, _mmatMatchInliers);
    LOG(INFO) << "inliers size: " <<  _mmatMatchInliers.rows;

    if( !success ||  _mmatMatchInliers.rows < 10){
        return false;
    }

    cv::Rodrigues(rvec, R);
    cv::cv2eigen(R, Reigen);
    cv::cv2eigen(tvec, teigen);
    _mseCorrectedCurrentPose = Sophus::SE3d(Reigen, teigen);

    // LOG(INFO) << "loop KF pose matrix: \n" << _mpLoopKF->Pose().matrix();
    // LOG(INFO) << "by pnp: \n" << _mseCorrectedCurrentPose.matrix();

    // // show the match result
    // cv::Mat img_goodmatch;
    // cv::drawMatches(_mpCurrentKF->mImageLeft, _mpCurrentKF->GetKeyPoints(), 
    //         _mpLoopKF->mImageLeft, _mpLoopKF->GetKeyPoints(), 
    //         _mvGoodFeatureMatches, img_goodmatch);
    // cv::resize(img_goodmatch, img_goodmatch, Size(), 0.5, 0.5);
    // cv::imshow("good matches", img_goodmatch);
    // cv::waitKey(1);

    // verify the reprojection error, as the inliers in solvePnPRansac() is not so reliable
    std::vector<cv::Point2f> vReprojectionPoints2d;
    cv::projectPoints(vLoopPoints3d, rvec, tvec, K, cv::Mat(), vReprojectionPoints2d);

    float disAverage = 0.0;

    //  show the reprojection result
    cv::Mat imgOut;
    cv::cvtColor(_mpCurrentKF->mImageLeft, imgOut,cv::COLOR_GRAY2RGB);
    for(int i = 0; i < _mmatMatchInliers.rows; i++){
        // verify the reprojection error
        int index = _mmatMatchInliers.at<int>(i, 0);
        float xx = vCurrentPoints2d[index].x - vReprojectionPoints2d[index].x;
        float yy = vCurrentPoints2d[index].y - vReprojectionPoints2d[index].y;
        float dis = std::sqrt((xx * xx) + (yy * yy));
        disAverage += dis;

        cv::circle(imgOut, vCurrentPoints2d[index], 5, cv::Scalar(0, 0, 255), -1);
        cv::line(imgOut, vCurrentPoints2d[index], vReprojectionPoints2d[index], cv::Scalar(255, 0, 0), 2);
    }

    disAverage /=  _mmatMatchInliers.rows;
    if(disAverage > 5.991) {
        return false;
    }

    cv::imshow("output", imgOut);
    cv::waitKey(1);

    return true;
}

// -------------------------------------------------------------------------------------

void LoopClosing::LoopFusion(){

    // request the backend to pause, avoiding conflict
    auto pBackend = _mpBackend.lock();
    pBackend->RequestPause();
    while(! pBackend->IfHasPaused()){
        usleep(1000);
    }

    // avoid the conflict between frontend tracking and loopclosing correction
    std::unique_lock<std::mutex> lck(_mpMap->mmutexMapUpdate);

    // correct the KFs and mappoints in the active map

    std::unordered_map<unsigned long, SE3> correctedActivePoses;
    correctedActivePoses.insert({_mpCurrentKF->mnKFId, _mseCorrectedCurrentPose});

    // calculate the relative pose between current KF and KFs in active map
    // and insert to the correctedActivePoses map
    for(auto &keyframe: _mpMap->GetActiveKeyFrames()){
        unsigned long kfId = keyframe.first;
        if(kfId == _mpCurrentKF->mnKFId){
            continue;
        }
        SE3 Tac = keyframe.second->Pose() * (_mpCurrentKF->Pose().inverse());
        SE3 Ta_corrected = Tac * _mseCorrectedCurrentPose;
        correctedActivePoses.insert({kfId, Ta_corrected});
    }

    // correct the active mappoints' positions
    for(auto &mappoint: _mpMap->GetActiveMapPoints()){
        MapPoint::Ptr mp = mappoint.second;

        assert(! mp->GetActiveObservations().empty());

        auto feat = mp->GetActiveObservations().front().lock();
        auto observingKF = feat->mpKF.lock();

        assert(correctedActivePoses.find(observingKF->mnKFId) != 
                correctedActivePoses.end());

        Vec3 posCamera = observingKF->Pose() * mp->Pos();
        SE3 Ta_corrected = correctedActivePoses.at(observingKF->mnKFId);
        mp->SetPos(Ta_corrected.inverse() * posCamera);
    }

    // correct the active KFs' poses
    for(auto &keyframe: _mpMap->GetActiveKeyFrames()){
        keyframe.second->SetPose(correctedActivePoses.at(keyframe.first));
    }

     // replace the current KF's mappoints with loop KF's matched mappoints
    for(int i = 0; i < _mmatMatchInliers.rows; i++){
        int index = _mmatMatchInliers.at<int>(i, 0);
        cv::DMatch match = _mvGoodFeatureMatches[index];
        auto loop_mp = _mpLoopKF->mvpFeaturesLeft[match.trainIdx]->mpMapPoint.lock();
        
        if(loop_mp){
            auto current_mp = _mpCurrentKF->mvpFeaturesLeft[match.queryIdx]->mpMapPoint.lock();
            if(current_mp){
                for(auto &obs: current_mp->GetObservations()){
                    auto obs_feat = obs.lock();
                    loop_mp->AddObservation(obs_feat);
                    obs_feat->mpMapPoint = loop_mp;
                }
                for(auto &obs: current_mp->GetActiveObservations()){
                    loop_mp->AddActiveObservation(obs.lock());
                }
                _mpMap->RemoveMapPoint(current_mp);
            }else{
                _mpCurrentKF->mvpFeaturesLeft[match.queryIdx]->mpMapPoint = loop_mp;
            }            
        }
    }

    _mpLastClosedKF = _mpCurrentKF;

    // resume the backend
    pBackend->Resume();

    LOG(INFO) << "corrected the map: correct current KF " << _mpCurrentKF->mnKFId << " and other active KFs";
}




// -------------------------------------------------------------------------------------

void LoopClosing::AddToDatabase(){
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
        || pNewKF->mnKFId - _mpLastClosedKF->mnKFId > 5){
            _mlNewKeyFrames.push_back(pNewKF);
        }
}

// ---------------------------------------------------------------------------------------








}   // namespace myslam