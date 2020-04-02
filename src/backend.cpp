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
    _mbHasPaused.store(false);

    // launch the backend thread
    _mthreadBackend = std::thread(std::bind(&Backend::BackendRun, this));
}


// -----------------------------------------------------------------------------------

void Backend::InsertKeyFrame(KeyFrame::Ptr pKF){
    std::unique_lock<std::mutex> lck(_mmutexNewKF);
    _mlNewKeyFrames.push_back(pKF);

    // need active map optimization when there is a new KF inserted
    _mbNeedOptimization = true;
    // UpdateMap();
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
    // _mapUpdate.notify_one();
    _mthreadBackend.join();
}

// -----------------------------------------------------------------------------------

void Backend::BackendRun(){

    while(_mbBackendIsRunning.load()){

        // std::unique_lock<std::mutex> lck(_mmutexData);
        // _mapUpdate.wait(lck);

        // process all new KFs until the new KF list is empty
        while(CheckNewKeyFrames()){
            ProcessNewKeyFrame();
        }

        // if the loopclosing thread asks backend to pause
        // this will make sure that the backend will pause in this position, having processed all new KFs in the list
        while(_mbRequestPause.load()){
            _mbHasPaused.store(true);
            usleep(1000);
        }
        _mbHasPaused.store(false);

        // optimize the active KFs and mappoints
        if(!CheckNewKeyFrames() && _mbNeedOptimization){
            OptimizeActiveMap();
            _mbNeedOptimization = false;  // this will become true when next new KF is inserted
        }
        
        usleep(1000);
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

    Map::KeyFramesType activeKFs = _mpMap->GetActiveKeyFrames();
    Map::MapPointsType activeMPs = _mpMap->GetActiveMapPoints();

    // add keyframe vertices
    std::unordered_map<unsigned long, VertexPose *> vertices_kfs;
    unsigned long maxKFId = 0;
    for(auto &keyframe: activeKFs){
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
    int index = 1; // edge index
    double chi2_th = 5.991;

    // add mappoint vertices
    // add edges
    std::unordered_map<unsigned long, VertexXYZ *> vertices_mps;
    std::unordered_map<EdgeProjection *, Feature::Ptr> edgesAndFeatures;
    for(auto &mappoint: activeMPs){
        auto mp = mappoint.second;
        if(mp->mbIsOutlier) {
            continue;
        }
            
        unsigned long mappointId = mp->mnId;
        VertexXYZ *v = new VertexXYZ;
        v->setEstimate(mp->Pos());
        v->setId((maxKFId +1) + mappointId);
        v->setMarginalized(true);

        // if the KF which first observes this mappoint is not in the active map,
        // then fixed this mappoint (be included just as constraint)
        if(activeKFs.find(mp->GetObservations().front().lock()->mpKF.lock()->mnKFId) == activeKFs.end()){
            v->setFixed(true);
        }

        vertices_mps.insert({mappointId, v});
        optimizer.addVertex(v);

        // edges
        for(auto &obs: mp->GetActiveObservations()){
            auto feat = obs.lock();
            auto kf = feat->mpKF.lock();

            assert(activeKFs.find(kf->mnKFId) != activeKFs.end());

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
            if(ef.first->chi2() > chi2_th){
                cntOutlier++;
            }else{
                cntInlier++;
            }
        }
        double inlierRatio = cntInlier / double(cntInlier + cntOutlier);
        if(inlierRatio > 0.5){
            break;
        }else{
            // chi2_th *= 2;
            iteration++;
        }
    }

    // process the outlier edges
    // remove the link between the feature and the mappoint
    for(auto &ef: edgesAndFeatures){
        if(ef.first->chi2() > chi2_th){
            ef.second->mbIsOutlier = true;
            auto mappoint = ef.second->mpMapPoint.lock();
            mappoint->RemoveActiveObservation(ef.second);
            mappoint->RemoveObservation(ef.second);
            // if the mappoint has no good observation, then regard it as a outlier. It will be deleted later.
            if(mappoint->GetObservations().empty()){
                mappoint->mbIsOutlier = true;
                _mpMap->AddOutlierMapPoint(mappoint->mnId);
            }
            ef.second->mpMapPoint.reset();
        }else{
            ef.second->mbIsOutlier = false;
        }
    }

    { // mutex
        std::unique_lock<std::mutex> lck(_mpMap->mmutexMapUpdate);
        // update the pose and landmark position
        for (auto &v: vertices_kfs) {
            activeKFs.at(v.first)->SetPose(v.second->estimate());
        }
        for (auto &v: vertices_mps){
            activeMPs.at(v.first)->SetPos(v.second->estimate());
        }

        // delete outlier mappoints
        _mpMap->RemoveAllOutlierMapPoints();
        _mpMap->RemoveOldActiveMapPoints();
    } // mutex

     // LOG(INFO) << "Backend: Outlier/Inlier in optimization: " << cntOutlier << "/" << cntInlier;
}

// ------------------------------------------------------------------------------









} // namespace myslam