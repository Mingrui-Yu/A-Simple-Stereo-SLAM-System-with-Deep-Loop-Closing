#include "myslam/backend.h"

#include "myslam/map.h"
#include "myslam/keyframe.h"
#include "myslam/mappoint.h"
#include "myslam/feature.h"
#include "myslam/g2o_types.h"
#include "myslam/camera.h"
#include "myslam/algorithm.h"

namespace myslam{

// -----------------------------------------------------------------------------------
Backend::Backend(){
    _mbBackendIsRunning.store(true);
    _mthreadBackend = std::thread(std::bind(&Backend::BackendLoop, this));
}

// -----------------------------------------------------------------------------------

void Backend::UpdateMap(){
    std::unique_lock<std::mutex> lck(_mmutexData);
    _mapUpdate.notify_one();
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
        std::unique_lock<std::mutex> lck(_mmutexData);
        _mapUpdate.wait(lck);
        LOG(INFO) << "start backend update the map.";

        // optimize the active KFs and mappoints
        Map::KeyFramesType activeKFs = _mpMap->GetActiveKeyFrames();
        Map::MapPointsType activeMPs = _mpMap->GetActiveMapPoints();
        LOG(INFO) << "start backend optimization.";
        OptimizeActiveMap(activeKFs, activeMPs);
    }
}

// -----------------------------------------------------------------------------------

void Backend::OptimizeActiveMap(Map::KeyFramesType &keyframes, Map::MapPointsType &mappoints){
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(
        g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // add keyframe vertex
    std::map<unsigned long, VertexPose *> vertices_kfs;
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

    // add mappoints vertex
    // add edges
    std::map<unsigned long, VertexXYZ *> vertices_mps;
    std::map<EdgeProjection *, Feature::Ptr> edgesAndFeatures;
    for(auto &mappoint: mappoints){
        auto mp = mappoint.second;
        if(mp->mbIsOutlier) 
            continue;
        unsigned long mappointId = mp->mnId;
        VertexXYZ *v = new VertexXYZ;
        v->setEstimate(mp->Pos());
        v->setId((maxKFId +1) + mappointId);
        v->setMarginalized(true);
        vertices_mps.insert({mappointId, v});
        optimizer.addVertex(v);


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

    LOG(INFO) << "do optimization.";

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
        LOG(INFO) << "inlierRatio: " << inlierRatio;
        if(inlierRatio > 0.5){
            break;
        }else{
            chi2_th *= 2;
            iteration++;
        }
    }
    for(auto &ef: edgesAndFeatures){
        if(ef.first->chi2() > chi2_th){
            ef.second->mbIsOutlier = true;
            ef.second->mpMapPoint.lock()->RemoveActiveObservation(ef.second);
        }else{
            ef.second->mbIsOutlier = false;
        }
    }
    LOG(INFO) << "Outlier/Inlier in backend optimization: " << cntOutlier << "/" << cntInlier;

    // Set pose and landmark position
    for (auto &v: vertices_kfs) {
        keyframes.at(v.first)->SetPose(v.second->estimate());
    }
    for (auto &v: vertices_mps){
        mappoints.at(v.first)->SetPos(v.second->estimate());
    }
}

// ------------------------------------------------------------------------------









} // namespace myslam