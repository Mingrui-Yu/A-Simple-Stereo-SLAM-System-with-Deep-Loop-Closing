#include "myslam/backend.h"

#include "myslam/map.h"
#include "myslam/frame.h"
#include "myslam/keyframe.h"
#include "myslam/mappoint.h"
#include "myslam/feature.h"
#include "myslam/g2o_types.h"
#include "myslam/camera.h"
#include "myslam/algorithm.h"
#include "myslam/viewer.h"
#include "myslam/loopclosing.h"

namespace myslam{

// -----------------------------------------------------------------------------------
Backend::Backend(){
    _mbBackendIsRunning.store(true);
    _mbRequestPause.store(false);
    _mthreadBackend = std::thread(std::bind(&Backend::BackendLoop, this));
}


// -----------------------------------------------------------------------------------

void Backend::InsertKeyFrame(KeyFrame::Ptr pKF){
    std::unique_lock<std::mutex> lck(_mmutexNewKF);
    _mlNewKeyFrames.push_back(pKF);
    _mbNeedOptimization = true;
}

// -----------------------------------------------------------------------------------

// void Backend::UpdateMap(){
//     std::unique_lock<std::mutex> lck(_mmutexData);
//     _mapUpdate.notify_one();
// }

// -----------------------------------------------------------------------------------

void Backend::RequestPause(){
    _mbRequestPause.store(true); 
}

// -----------------------------------------------------------------------------------

bool Backend::IfHasPaused(){
    return (_mbRequestPause.load()) && (_mbHasPaused.load());
}

// -----------------------------------------------------------------------------------

void Backend::Resume(){
    _mbRequestPause.store(false); 
}

// -----------------------------------------------------------------------------------

void Backend::Stop(){
    _mbBackendIsRunning.store(false);
    _mapUpdate.notify_one();
    _mthreadBackend.join();
}

// -----------------------------------------------------------------------------------

void Backend::BackendLoop(){

    while(_mbBackendIsRunning.load()){

        while(CheckNewKeyFrames()){
            ProcessNewKeyFrame();
        }

        usleep(1000);

        while(_mbRequestPause.load()){
            _mbHasPaused.store(true);
            usleep(1000);
        }
        _mbHasPaused.store(false);

        // optimize the active KFs and mappoints
        if(!CheckNewKeyFrames() && _mbNeedOptimization){
            OptimizeActiveMap();
            _mbNeedOptimization = false;  // until the next inserted KF, this will become true
        }
        
    }

}

// -----------------------------------------------------------------------------------

bool Backend::CheckNewKeyFrames(){
    std::unique_lock<std::mutex> lck(_mmutexNewKF);
    return(!_mlNewKeyFrames.empty());
}

// -----------------------------------------------------------------------------------
void Backend::ProcessNewKeyFrame(){
    {
        std::unique_lock<std::mutex> lck(_mmutexNewKF);
        _mpCurrentKF = _mlNewKeyFrames.front();
        _mlNewKeyFrames.pop_front();
    }
    
    _mpMap->InsertKeyFrame(_mpCurrentKF);
    _mpLoopClosing->InsertNewKeyFrame(_mpCurrentKF);

    // if(_mpViewer){
    //     _mpViewer->UpdateMap();
    // }

    // _mpLastKF = _mpCurrentKF;
}

// -----------------------------------------------------------------------------------

void Backend::OptimizeActiveMap(){
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(
        g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    Map::KeyFramesType keyframes = _mpMap->GetActiveKeyFrames();
    Map::MapPointsType mappoints = _mpMap->GetActiveMapPoints();

    // for(auto &keyframe: keyframes){
    //     LOG(INFO) << "active keyframe id: " << keyframe.second->mnKFId;
    // }

    // add keyframe vertex
    std::unordered_map<unsigned long, VertexPose *> vertices_kfs;
    unsigned long maxKFId = 0;
    for(auto &keyframe: keyframes){
        auto kf = keyframe.second;
        VertexPose *vertex_pose = new VertexPose();
        vertex_pose->setId(kf->mnKFId);
        vertex_pose->setEstimate(kf->Pose());
        optimizer.addVertex(vertex_pose);

        maxKFId = std::max(maxKFId, kf->mnKFId);
        vertices_kfs.insert({kf->mnKFId, vertex_pose});

    }

    Mat33 camK = _mpCameraLeft->K();
    SE3 camExt = _mpCameraLeft->Pose();
    int index = 1;
    double chi2_th = 5.991;


    int cntFixedMP = 0;

    // add mappoints vertex
    // add edges
    std::unordered_map<unsigned long, VertexXYZ *> vertices_mps;
    std::unordered_map<EdgeProjection *, Feature::Ptr> edgesAndFeatures;
    for(auto &mappoint: mappoints){
        auto mp = mappoint.second;
        if(mp->mbIsOutlier) {
            continue;
        }
            
        unsigned long mappointId = mp->mnId;
        VertexXYZ *v = new VertexXYZ;
        v->setEstimate(mp->Pos());
        v->setId((maxKFId +1) + mappointId);
        v->setMarginalized(true);

        if(keyframes.find(mp->GetObservations().front().lock()->mpKF.lock()->mnKFId) == keyframes.end()){
            v->setFixed(true);
            // LOG(INFO) << "fixed mappoint id: " << mp->mnId;
            cntFixedMP++;
        }

        vertices_mps.insert({mappointId, v});
        optimizer.addVertex(v);

        // LOG(INFO) << "mappoint's id: " << mp->mnId;
        // assert(_mpMap->GetAllMapPoints().find(mp->mnId) != _mpMap->GetAllMapPoints().end());
        // for(auto &obs: mp->GetActiveObservations()){
        //     auto feat = obs.lock();
        //     auto kf = feat->mpKF.lock();
        //     LOG(INFO) << "observing kf id: " << kf->mnKFId;
        // }


        for(auto &obs: mp->GetActiveObservations()){
            auto feat = obs.lock();
            auto kf = feat->mpKF.lock();

            assert(feat->mbIsOnLeftFrame == true);
            assert(keyframes.find(kf->mnKFId) != keyframes.end());

            if(feat->mbIsOutlier)
                continue;
            EdgeProjection *edge = new EdgeProjection(camK, camExt);
            edge->setId(index);
            edge->setVertex(0, vertices_kfs.at(kf->mnKFId));
            edge->setVertex(1, vertices_mps.at(mp->mnId));
            edge->setMeasurement(toVec2(feat->mkpPosition.pt));
            edge->setInformation(Mat22::Identity());
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(chi2_th);
            edge->setRobustKernel(rk);
            edgesAndFeatures.insert({edge, feat});

            optimizer.addEdge(edge);
            index++;
        }
    }
    // LOG(INFO) << "all mappoiints number: " << mappoints.size();
    // LOG(INFO) << "fixed mappoints number: " << cntFixedMP;

    // do optimization
    int cntOutlier = 0, cntInlier = 0;
    int iteration = 0;

    while(iteration < 5){
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cntOutlier = 0;
        cntInlier = 0;
        // determine if we want to adjust the outlier threshold
        for(auto &ef: edgesAndFeatures){
            if(ef.first->chi2() > chi2_th){ // chi2_th
                cntOutlier++;
            }else{
                cntInlier++;
            }
        }
        double inlierRatio = cntInlier / double(cntInlier + cntOutlier);
        // LOG(INFO) << "inlierRatio: " << inlierRatio;
        if(inlierRatio > 0.5){
            break;
        }else{
            // chi2_th *= 2;
            iteration++;
        }
    }

    for(auto &ef: edgesAndFeatures){
        if(ef.first->chi2() > chi2_th){
            ef.second->mbIsOutlier = true;
            auto mappoint = ef.second->mpMapPoint.lock();
            mappoint->RemoveActiveObservation(ef.second);
            mappoint->RemoveObservation(ef.second);
            if(mappoint->GetObservations().empty()){
                mappoint->mbIsOutlier = true;
            }
            ef.second->mpMapPoint.reset();
        }else{
            ef.second->mbIsOutlier = false;
        }
    }
    LOG(INFO) << "Backend: Outlier/Inlier in optimization: " << cntOutlier << "/" << cntInlier;

    // Set pose and landmark position
    for (auto &v: vertices_kfs) {
        keyframes.at(v.first)->SetPose(v.second->estimate());
    }
    for (auto &v: vertices_mps){
        mappoints.at(v.first)->SetPos(v.second->estimate());
    }

    // delete outlier mappoints
     for(auto &mappoint: mappoints){
        auto mp = mappoint.second;
        if(mp->mbIsOutlier) {
            _mpMap->RemoveMapPoint(mp);
        }
     }
     _mpMap->RemoveOldActiveMapPoints();
}

// ------------------------------------------------------------------------------









} // namespace myslam