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
#include "myslam/g2o_types.h"

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

    nLevels = Config::Get<int>("ORBextractor.nLevels");

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
    {
        std::unique_lock<std::mutex> lck(_mmutexNewKFList);
        _mpCurrentKF = _mlNewKeyFrames.front();
        _mlNewKeyFrames.pop_front();
    }

    // calculate the whole image's descriptor vector with deeplcd
    _mpCurrentKF->mpDescrVector = _mpDeepLCD->calcDescrOriginalImg(_mpCurrentKF->mImageLeft);
    
    std::vector<cv::KeyPoint> vPyramidKPs;
    vPyramidKPs.reserve(nLevels * _mpCurrentKF->mvpFeaturesLeft.size());
    
    for(size_t i = 0, N = _mpCurrentKF->mvpFeaturesLeft.size(); i < N; i++){
        for(int level = 0; level < nLevels; level++){
            _mpCurrentKF->mvpFeaturesLeft[i]->mkpPosition.class_id = i;
            cv::KeyPoint kp(_mpCurrentKF->mvpFeaturesLeft[i]->mkpPosition);
            kp.octave = level;
            kp.response = -1;
            kp.class_id = i;
            vPyramidKPs.push_back(kp);
        }
    }

    _mpORBextractor->ScreenAndComputeKPsParams(_mpCurrentKF->mImageLeft, 
        vPyramidKPs, _mpCurrentKF->mvPyramidKeyPoints);
    
    // LOG(INFO) << "kps size: " << vPyramidKPs.size() << "  " << _mpCurrentKF->mvPyramidKeyPoints.size();

    // calculate the orb descriptors of the keypoints using function in "myslam/ORBextractor.h"
    _mpORBextractor->CalcDescriptors(_mpCurrentKF->mImageLeft, 
            _mpCurrentKF->mvPyramidKeyPoints, _mpCurrentKF->mORBDescriptors);

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

    _mpMatcher->match(_mpCurrentKF->mORBDescriptors, _mpLoopKF->mORBDescriptors, matches);
    
    auto min_max = std::minmax_element(matches.begin(), matches.end(),
                [] (const cv::DMatch &m1, const cv::DMatch &m2) {return m1.distance < m2.distance;});
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    LOG(INFO) << "--Max dist = " << max_dist;
    LOG(INFO) << "--Min dist = " << min_dist;

    _msetValidFeatureMatches.clear();   
    for (auto &match: matches){
        if(match.distance <= max(2*min_dist, 30.0)){
            int loopFeatureId = _mpLoopKF->mvPyramidKeyPoints[match.trainIdx].class_id;
            int currentFeatureId = _mpCurrentKF->mvPyramidKeyPoints[match.queryIdx].class_id;

            if(_msetValidFeatureMatches.find({currentFeatureId, loopFeatureId}) 
                    != _msetValidFeatureMatches.end()){
                continue;
            }
            _msetValidFeatureMatches.insert({currentFeatureId, loopFeatureId});
        }
    }

    LOG(INFO) << "match number: " << matches.size() ;
    LOG(INFO) << "good match number: " << _msetValidFeatureMatches.size();

    
    if(_msetValidFeatureMatches.size() < 10){
        return false;
    }

    return true;
}


// ----------------------------------------------------------------------------------

bool LoopClosing::ComputeSE3(){

    // prepare the data for PnP solver
    std::vector<cv::Point3f> vLoopPoints3d;
    std::vector<cv::Point2f> vCurrentPoints2d;
    std::vector<cv::Point2f> vLoopPoints2d;
    std::vector<cv::DMatch> vMatchesWithMapPoint;

    for(auto iter = _msetValidFeatureMatches.begin(); iter != _msetValidFeatureMatches.end(); ){
        int currentFeatureId = (*iter).first;
        int loopFeatureId = (*iter).second;
        auto mp = _mpLoopKF->mvpFeaturesLeft[loopFeatureId]->mpMapPoint.lock();
        if(mp){
            vCurrentPoints2d.push_back(
                _mpCurrentKF->mvpFeaturesLeft[currentFeatureId]->mkpPosition.pt);
            Vec3 pos = mp->Pos();
            vLoopPoints3d.push_back(cv::Point3f(pos(0), pos(1), pos(2)));
            vLoopPoints2d.push_back(_mpLoopKF->mvpFeaturesLeft[loopFeatureId]->mkpPosition.pt);
            
            // useful if needs to draw the matches
            cv::DMatch valid_match(currentFeatureId, loopFeatureId, 10.0);
            vMatchesWithMapPoint.push_back(valid_match);

            iter++;
        }else{
            iter = _msetValidFeatureMatches.erase(iter);
        }
    }

    //  //  show the match result
    // cv::Mat img_goodmatch;
    // cv::drawMatches(_mpCurrentKF->mImageLeft, _mpCurrentKF->GetKeyPoints(), 
    //         _mpLoopKF->mImageLeft, _mpLoopKF->GetKeyPoints(), 
    //         vMatchesWithMapPoint, img_goodmatch);
    // cv::resize(img_goodmatch, img_goodmatch, Size(), 0.5, 0.5);
    // cv::imshow("2", img_goodmatch);
    // cv::waitKey(10);

    LOG(INFO) << "number of valid 3d-2d pairs: " << vLoopPoints3d.size();

    if(vLoopPoints3d.size() < 10) 
        return false;

    cv::Mat rvec, tvec, R, K;
    cv::eigen2cv(_mpCameraLeft->K(), K);

    Eigen::Matrix3d Reigen;
    Eigen::Vector3d teigen;
    _mmatMatchInliers.release();

    try{
        cv::solvePnPRansac(vLoopPoints3d, vCurrentPoints2d, 
            K, cv::Mat(), rvec, tvec, false, 100, 5.991, 0.99, _mmatMatchInliers);
    }catch(...){
        return false;
    }
    

    cv::Rodrigues(rvec, R);
    cv::cv2eigen(R, Reigen);
    cv::cv2eigen(tvec, teigen);
    _mseCorrectedCurrentPose = Sophus::SE3d(Reigen, teigen);

    int cntInliers = OptimizeCurrentPose();

    LOG(INFO) << "inliers size (after optimization): " <<  cntInliers;

    if( cntInliers < 10){
        return false;
    }

    double error = (_mpCurrentKF->Pose() * _mseCorrectedCurrentPose.inverse()).log().norm();
    if (error > 1){
        _mbNeedCorrect = true;
    } else{
        _mbNeedCorrect = false;
    }

    // verify the reprojection error
    Vec3 t_eigen = _mseCorrectedCurrentPose.translation();
    Mat33 R_eigen = _mseCorrectedCurrentPose.rotationMatrix();
    cv::Mat R_cv, t_cv, r_cv;
    cv::eigen2cv(R_eigen, R_cv);
    cv::eigen2cv(t_eigen, t_cv);
    cv::Rodrigues(R_cv, r_cv);
    std::vector<cv::Point2f> vReprojectionPoints2d;
    std::vector<cv::Point2f> vCurrentKeyPoints;
    std::vector<cv::Point3f> vLoopMapPoints;
    for(auto iter = _msetValidFeatureMatches.begin(); iter != _msetValidFeatureMatches.end(); iter++){
        int currentFeatureId = (*iter).first;
        int loopFeatureId = (*iter).second;
        auto mp = _mpLoopKF->mvpFeaturesLeft[loopFeatureId]->mpMapPoint.lock();
        vCurrentKeyPoints.push_back(
            _mpCurrentKF->mvpFeaturesLeft[currentFeatureId]->mkpPosition.pt);
        Vec3 pos = mp->Pos();
        vLoopMapPoints.push_back(cv::Point3f(pos(0), pos(1), pos(2)));
    }
    cv::projectPoints(vLoopMapPoints, r_cv, t_cv, K, cv::Mat(), vReprojectionPoints2d);

    //  show the reprojection result
    float disAverage = 0.0;
    cv::Mat imgOut;
    cv::cvtColor(_mpCurrentKF->mImageLeft, imgOut,cv::COLOR_GRAY2RGB);
    for(size_t i = 0, N = vLoopMapPoints.size(); i < N; i++){
        // verify the reprojection error
        int index = i;
        float xx = vCurrentKeyPoints[index].x - vReprojectionPoints2d[index].x;
        float yy = vCurrentKeyPoints[index].y - vReprojectionPoints2d[index].y;
        float dis = std::sqrt((xx * xx) + (yy * yy));
        disAverage += dis;

        cv::circle(imgOut, vCurrentKeyPoints[index], 5, cv::Scalar(0, 0, 255), -1);
        cv::line(imgOut, vCurrentKeyPoints[index], vReprojectionPoints2d[index], cv::Scalar(255, 0, 0), 2);
    }

    disAverage /=  vLoopMapPoints.size();

    LOG(INFO) << "average reprojection error: " << disAverage;

    cv::imshow("output", imgOut);
    cv::waitKey(10);

    if(disAverage > 5.991) {
        return false;
    }

    // now has passed all verification, regard it as the true loop keyframe
    _mpCurrentKF->mpLoopKF = _mpLoopKF;
    _mpCurrentKF->mRelativePoseToLoopKF = 
            _mseCorrectedCurrentPose * _mpLoopKF->Pose().inverse();
    _mpLastClosedKF = _mpCurrentKF;


    return true;
}

// ------------------------------------------------------------------------------------------

int LoopClosing::OptimizeCurrentPose(){

    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    //vertex
    VertexPose *vertex_pose = new VertexPose();
    vertex_pose->setId(0);
    vertex_pose->setEstimate(_mseCorrectedCurrentPose);
    optimizer.addVertex(vertex_pose);

    Mat33 K = _mpCameraLeft->K();

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<bool> vEdgeIsOutlier;
    vEdgeIsOutlier.reserve(_msetValidFeatureMatches.size());
    edges.reserve(_msetValidFeatureMatches.size());
    std::vector<std::pair<int, int> > vMatches;
    vMatches.reserve(_msetValidFeatureMatches.size());
    for(auto iter = _msetValidFeatureMatches.begin(); iter != _msetValidFeatureMatches.end(); iter++){
        int currentFeatureId = (*iter).first;
        int loopFeatureId = (*iter).second;
        auto mp = _mpLoopKF->mvpFeaturesLeft[loopFeatureId]->mpMapPoint.lock();
        auto point2d = _mpCurrentKF->mvpFeaturesLeft[currentFeatureId]->mkpPosition.pt;
        assert(mp != nullptr);

        EdgeProjectionPoseOnly *edge = new EdgeProjectionPoseOnly(mp->Pos(), K);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(toVec2(point2d));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber);
        edges.push_back(edge);
        vEdgeIsOutlier.push_back(false);
        vMatches.push_back(*iter);
        optimizer.addEdge(edge);
        index++;
    }

    // estimate the Pose and determine the outliers
    // start optimization
    const double chi2_th = 5.991;
    int cntOutliers = 0;
    int numIterations = 4;

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    for(int iteration = 0; iteration < numIterations; iteration++){
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cntOutliers = 0;

        // count the outliers
        for(size_t i = 0, N = edges.size(); i < N; i++){
            auto e = edges[i];
            if(vEdgeIsOutlier[i]){
                e->computeError();
            }
            if(e->chi2() > chi2_th){
                vEdgeIsOutlier[i] = true;
                e->setLevel(1);
                cntOutliers++;
            } else{
                vEdgeIsOutlier[i] = false;
                e->setLevel(0);
            }

            if(iteration == numIterations - 2){
                e->setRobustKernel(nullptr);
            }
        }
    }

    // remove the outlier match
    for(size_t i = 0, N = vEdgeIsOutlier.size(); i < N; i++){
        if(vEdgeIsOutlier[i]){
            _msetValidFeatureMatches.erase(vMatches[i]);
        }
    }
   
    _mseCorrectedCurrentPose = vertex_pose->estimate();

    return _msetValidFeatureMatches.size();
}

// -------------------------------------------------------------------------------------

void LoopClosing::LoopFusion(){

    if( ! _mbNeedCorrect){
        return;
    }

    // request the backend to pause, avoiding conflict
    auto pBackend = _mpBackend.lock();
    pBackend->RequestPause();
    while(! pBackend->IfHasPaused()){
        usleep(1000);
    }

    LOG(INFO) << "Backend pause." ;

    // correct the KFs and mappoints in the active map

    { // mutex
        // avoid the conflict between frontend tracking and loopclosing correction
        std::unique_lock<std::mutex> lck(_mpMap->mmutexMapUpdate);

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
            // LOG(INFO) << "correct mappoint id: " << mappoint.first;
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
        for(auto iter = _msetValidFeatureMatches.begin(); iter != _msetValidFeatureMatches.end(); iter++){
             int currentFeatureId = (*iter).first;
            int loopFeatureId = (*iter).second;

            auto loop_mp = _mpLoopKF->mvpFeaturesLeft[loopFeatureId]->mpMapPoint.lock();
            assert(loop_mp != nullptr);

            if(loop_mp){
                auto current_mp = _mpCurrentKF->mvpFeaturesLeft[currentFeatureId]->mpMapPoint.lock();
                if(current_mp){

                    LOG(INFO) << "current mp id: " << current_mp->mnId;
                    LOG(INFO) << "loop mp id: " << loop_mp->mnId;

                    for(auto &obs: current_mp->GetObservations()){
                        auto obs_feat = obs.lock();
                        loop_mp->AddObservation(obs_feat);
                        obs_feat->mpMapPoint = loop_mp;
                    }
                    // for(auto &obs: current_mp->GetActiveObservations()){
                    //     // obs.lock()->mpMapPoint = loop_mp;
                    //     loop_mp->AddActiveObservation(obs.lock());
                    // }

                    // LOG(INFO) << "current mp count(): " <<  current_mp.use_count(); 

                    // _mpMap->InsertActiveMapPoint(loop_mp);
                    _mpMap->RemoveMapPoint(current_mp);

                    // assert(_mpMap->GetAllMapPoints().find(current_mp->mnId) == _mpMap->GetAllMapPoints().end());
                    // assert(_mpMap->GetActiveMapPoints().find(current_mp->mnId) == _mpMap->GetActiveMapPoints().end());
                    
                    // LOG(INFO) << "current mp count(): " <<  current_mp.use_count(); 
                    // _mpCurrentKF->mvpFeaturesLeft[currentFeatureId]->mpMapPoint = loop_mp;
                }else{
                    _mpCurrentKF->mvpFeaturesLeft[currentFeatureId]->mpMapPoint = loop_mp;
                }            
            }
        }

    } // mutex
   
    PoseGraphOptimization();

    // resume the backend
    pBackend->Resume();

    LOG(INFO) << "corrected the map: correct current KF " << _mpCurrentKF->mnKFId << " and other active KFs";
}

// -------------------------------------------------------------------------------------------

void LoopClosing::PoseGraphOptimization(){
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; 
    optimizer.setAlgorithm(solver); 

    Map::KeyFramesType keyframes = _mpMap->GetAllKeyFrames();

    std::map<unsigned long, VertexPose *> vertices_kf; 
    for(auto &keyframe: keyframes){
        unsigned long kfId = keyframe.first;
        auto kf = keyframe.second;
        VertexPose *vertex_pose = new VertexPose();
        vertex_pose->setId(kf->mnKFId);
        vertex_pose->setEstimate(kf->Pose());
        vertex_pose->setMarginalized(false);

        auto mapActiveKFs = _mpMap->GetActiveKeyFrames();
        // active KFs, loop KF, initial KF are fixed
        if( kfId == _mpCurrentKF->mnKFId // mapActiveKFs.find(kfId) != mapActiveKFs.end()
                || (kfId == _mpLoopKF->mnKFId) || kfId == 0){
            vertex_pose->setFixed(true);
        }

        optimizer.addVertex(vertex_pose);
        vertices_kf.insert({kf->mnKFId, vertex_pose});
    }

    int index = 0;
    std::map<int, EdgePoseGraph *> vEdges;
    for(auto &keyframe: keyframes){
        unsigned long kfId = keyframe.first;
        assert(vertices_kf.find(kfId) != vertices_kf.end());
        auto kf = keyframe.second;

        auto lastKF = kf->mpLastKF.lock();
        if(lastKF){
            EdgePoseGraph *edge = new EdgePoseGraph();
            edge->setId(index);
            edge->setVertex(0, vertices_kf.at(kfId));

            // LOG(INFO) << "kfid: " << kfId << "  last kf id: " << lastKF->mnKFId;

            edge->setVertex(1, vertices_kf.at(lastKF->mnKFId));
            edge->setMeasurement(kf->mRelativePoseToLastKF);
            edge->setInformation(Mat66::Identity());

            optimizer.addEdge(edge);
            vEdges.insert({index, edge});
            
            index++;
        }
        auto loopKF = kf->mpLoopKF.lock();
        if(loopKF){
            EdgePoseGraph *edge = new EdgePoseGraph();
            edge->setId(index);

            // LOG(INFO) << "kfid: " << kfId << "  loop kf id: " << loopKF->mnKFId;

            edge->setVertex(0, vertices_kf.at(kfId));
            edge->setVertex(1, vertices_kf.at(loopKF->mnKFId));
            edge->setMeasurement(kf->mRelativePoseToLoopKF);
            edge->setInformation(Mat66::Identity());

            optimizer.addEdge(edge);
            vEdges.insert({index, edge});

            index++;
        }
    }

    LOG(INFO) << "set up done." ; 

    optimizer.initializeOptimization();
    optimizer.optimize(20);

    // for(auto &edge: vEdges){
    //     LOG(INFO) << "edge id: " << edge.first;
    //     LOG(INFO) << "edge error: " << edge.second->chi2();
    // }

    { // mutex
        // avoid the conflict between frontend tracking and loopclosing correction
        std::unique_lock<std::mutex> lck(_mpMap->mmutexMapUpdate);
        
        LOG(INFO) << "0";

        // set the mappoints' positions according to its first observing KF's optimized pose
        auto allMapPoints = _mpMap->GetAllMapPoints();
        auto activeMapPoints = _mpMap->GetActiveMapPoints();
        for(auto iter = activeMapPoints.begin(); iter != activeMapPoints.end(); iter++){
            allMapPoints.erase((*iter).first);
        }
        for(auto &mappoint: allMapPoints){
            MapPoint::Ptr mp = mappoint.second;

            if(mp->GetObservations().empty()){
                continue;
            }

            auto feat = mp->GetObservations().front().lock();
            auto observingKF = feat->mpKF.lock();

            Vec3 posCamera = observingKF->Pose() * mp->Pos();

            // LOG(INFO) << "observing kf id: " << observingKF->mnKFId;

            if(vertices_kf.find(observingKF->mnKFId) == vertices_kf.end()){
                continue;
            }

            SE3 T_optimized = vertices_kf.at(observingKF->mnKFId)->estimate();
            mp->SetPos(T_optimized.inverse() * posCamera);
        }

        // SE3 pose;
        // pose =_mpMap->GetAllKeyFrames().at(_mpCurrentKF->mnKFId - 10)->Pose();
        // LOG(INFO) << "pose before optimization: " << pose.matrix();

        // set the KFs' optimized poses
        for (auto &v: vertices_kf) {
            keyframes.at(v.first)->SetPose(v.second->estimate());
        }

        // pose =_mpMap->GetAllKeyFrames().at(_mpCurrentKF->mnKFId - 10)->Pose();
        // LOG(INFO) << "pose after optimization: " << pose.matrix();

        LOG(INFO) << "pose graph optimization done.";

    } // mutex

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
    } else{
        pNewKF->mImageLeft.release();
    }
}

// ---------------------------------------------------------------------------------------








}   // namespace myslam