#include "myslam/frontend.h"

#include "myslam/feature.h"
#include "myslam/frame.h"
#include "myslam/keyframe.h"
#include "myslam/mappoint.h"
#include "myslam/config.h"
#include "myslam/viewer.h"
#include "myslam/ORBextractor.h"
#include "myslam/camera.h"
#include "myslam/algorithm.h"
#include "myslam/map.h"
#include "myslam/g2o_types.h"
#include "myslam/backend.h"

namespace myslam{

// -------------------------------------------------------------


Frontend::Frontend(){
    _numFeatures = Config::Get<int>("num_features");
    _numFeaturesInit = Config::Get<int>("num_features_init");
    _numFeaturesTrackingGood = Config::Get<int>("num_features_tracking");
    _numFeaturesTrackingBad = Config::Get<int>("num_features_tracking_bad");
    _numFeaturesNeededForNewKF = Config::Get<int>("num_features_needed_for_keyframe");

    // int nFeatures =  Config::Get<int>("ORBextractor.nFeatures");
    float fScaleFactor = Config::Get<float>("ORBextractor.scaleFactor");
    int nLevels = Config::Get<int>("ORBextractor.nLevels");
    int fIniThFAST = Config::Get<int>("ORBextractor.iniThFAST");
    int fMinThFAST = Config::Get<int>("ORBextractor.minThFAST");

    // _orb = cv::ORB::create(_numFeatures, 1.2f, 8, 31, 0, 2,
    //     cv::ORB::HARRIS_SCORE, 31, 20);
    // _gftt = cv::GFTTDetector::create(_numFeatures, 0.01, 20);

    mpORBextractor = ORBextractor::Ptr(new ORBextractor(_numFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST));
}

// -------------------------------------------------------------

void Frontend::SetViewer(Viewer::Ptr viewer){
    _mpViewer = viewer;
}


// --------------------------------------------------------------

bool Frontend::GrabStereoImage (const cv::Mat &leftImg, const cv::Mat &rightImg, 
    const double &dTimeStamp){

    _mpCurrentFrame = Frame::Ptr(new Frame(leftImg, rightImg, dTimeStamp));

    // LOG(INFO) << "frame id: " << _mpCurrentFrame->mnFrameId;

    switch(_mStatus){
        case FrontendStatus::INITING:
        StereoInit();
        break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();
            break;
        case FrontendStatus::LOST:
            // Reset();
            break;
    }

    _mpLastFrame = _mpCurrentFrame;
    return true;
}


// -----------------------------------------------------------------------------

bool Frontend::Track(){
    if(_mpLastFrame){
        _mpCurrentFrame->SetPose(_mseRelativeMotion * (_mpLastFrame->Pose()));
    }

    TrackLastFrame();

    int numTrackingInliers = EstimateCurrentPose();
    if(numTrackingInliers > _numFeaturesTrackingGood){
        // tracking good
        _mStatus = FrontendStatus::TRACKING_GOOD;
    } else if (numTrackingInliers > _numFeaturesTrackingBad){
        // tracking bad
        _mStatus = FrontendStatus::TRACKING_BAD;
    } else{
        // lost
        _mStatus = FrontendStatus::LOST;
    }

    if (numTrackingInliers < _numFeaturesNeededForNewKF){
        InsertKeyFrame();
    }

    _mseRelativeMotion = _mpCurrentFrame->Pose() * _mpLastFrame->Pose().inverse();
    
    if(_mpViewer){
        _mpViewer->AddCurrentFrame(_mpCurrentFrame);
    }
    return true;
}


// -----------------------------------------------------------------------------

int Frontend::TrackLastFrame(){
     // use LK flow to estimate points betweem last frame and current frame
    std::vector<cv::Point2f> pointKpsLast, pointKpsCurrent;
    pointKpsLast.reserve(_mpLastFrame->mvpFeaturesLeft.size());
    pointKpsCurrent.reserve(_mpLastFrame->mvpFeaturesLeft.size());
    for(auto &feat: _mpLastFrame->mvpFeaturesLeft){
        auto mp = feat->mpMapPoint.lock();
        if(mp){
            auto px = _mpCameraLeft->world2pixel(mp->Pos(), _mpCurrentFrame->Pose());
            pointKpsLast.push_back(feat->mkpPosition.pt);
            pointKpsCurrent.push_back(cv::Point2f(px[0], px[1]));
        } else{
            pointKpsLast.push_back(feat->mkpPosition.pt);
            pointKpsCurrent.push_back(feat->mkpPosition.pt);
        }
    }
    // LOG(INFO) << "number of keypoints in last frame: " << pointKpsLast.size();

    std::vector<uchar> status;
    cv::Mat error;
    cv::calcOpticalFlowPyrLK(_mpLastFrame->mLeftImg, _mpCurrentFrame->mLeftImg,
        pointKpsLast, pointKpsCurrent, status, error, cv::Size(11, 11), 3, 
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    
    int numGoodPts = 0;
    _mpCurrentFrame->mvpFeaturesLeft.reserve(pointKpsCurrent.size());
    for(size_t i = 0, N = status.size(); i < N; i++){
        if(status[i]){
            cv::KeyPoint kp(pointKpsCurrent[i], 7);
            Feature::Ptr feature(new Feature(kp));
            feature->mpMapPoint = _mpLastFrame->mvpFeaturesLeft[i]->mpMapPoint;
            _mpCurrentFrame->mvpFeaturesLeft.push_back(feature);
            numGoodPts++;
        }
    }

    LOG(INFO) << "Find " << numGoodPts << " keypoints in the last frame.";
    return numGoodPts;
}


// --------------------------------------------------------------

int Frontend::EstimateCurrentPose(){
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
    vertex_pose->setEstimate(_mpCurrentFrame->Pose());
    optimizer.addVertex(vertex_pose);

    Mat33 K = _mpCameraLeft->K();

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<Feature::Ptr> features;
    features.reserve(_mpCurrentFrame->mvpFeaturesLeft.size());
    edges.reserve(_mpCurrentFrame->mvpFeaturesLeft.size());
    for(size_t i = 0, N = _mpCurrentFrame->mvpFeaturesLeft.size(); i < N; i++){
        auto mp = _mpCurrentFrame->mvpFeaturesLeft[i]->mpMapPoint.lock();
        if(mp){
            features.push_back(_mpCurrentFrame->mvpFeaturesLeft[i]);
            EdgeProjectionPoseOnly *edge = new EdgeProjectionPoseOnly(mp->Pos(), K);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(toVec2(_mpCurrentFrame->mvpFeaturesLeft[i]->mkpPosition.pt));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);

            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
       }
    }

    // estimate the Pose and determine the outliers
    // start optimization
    const double chi2_th = 5.991;
    int cntOutliers = 0;
    for(int iteration = 0; iteration < 4; iteration++){
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cntOutliers = 0;

        // count the outliers
        for(size_t i = 0, N = edges.size(); i < N; i++){
            auto e = edges[i];
            if(features[i]->mbIsOutlier){
                e->computeError();
            }
            if(e->chi2() > chi2_th){
                features[i]->mbIsOutlier = true;
                e->setLevel(1);
                cntOutliers++;
            } else{
                features[i]->mbIsOutlier = false;
                e->setLevel(0);
            }

            if(iteration == 2){
                e->setRobustKernel(nullptr);
            }
        }
    }
    LOG(INFO) << "Outliers/Inliers in frontend current pose estimating: " 
            << cntOutliers << "/" << features.size() - cntOutliers;

    // set pose and outlier
    _mpCurrentFrame->SetPose(vertex_pose->estimate());

    // remove the link beween outlier features and their corresponding mappoints
    for(auto &feat: features){
        if(feat->mbIsOutlier){
            feat->mpMapPoint.reset();
            feat->mbIsOutlier = false;
            // 是否可以考虑 删除 outlier 的mappoints?
        }
    }
    return features.size() - cntOutliers;
}



// --------------------------------------------------------------

bool Frontend::StereoInit(){
    DetectFeatures();
    int numCoorFeatures = FindFeaturesInRight();
    if (numCoorFeatures < _numFeaturesInit){
        return false;
    }
    bool bBuildMapSuccess = BuildInitMap();

    if(bBuildMapSuccess){
        _mStatus = FrontendStatus::TRACKING_GOOD;
        if(_mpViewer){
            _mpViewer->AddCurrentFrame(_mpCurrentFrame);
            _mpViewer->UpdateMap();
        }
        return true;
    }
    return false;
}




// --------------------------------------------------------------

int Frontend::DetectFeatures(){
    cv::Mat mask(_mpCurrentFrame->mLeftImg.size(), CV_8UC1, 255);
    for (auto &feat: _mpCurrentFrame->mvpFeaturesLeft){
        cv::rectangle(mask, feat->mkpPosition.pt - cv::Point2f(10, 10),
            feat->mkpPosition.pt + cv::Point2f(10, 10), 0, CV_FILLED);
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    mpORBextractor->Detect(_mpCurrentFrame->mLeftImg, mask, keypoints);
    int cntDetected = 0;
    for(auto &kp: keypoints) {
        _mpCurrentFrame->mvpFeaturesLeft.push_back(
            Feature::Ptr(new Feature(kp)));
        cntDetected++;
    }
    LOG(INFO) << "Detect " << cntDetected << " new features in current frame";
    return cntDetected;
}




// --------------------------------------------------------------

int Frontend::FindFeaturesInRight(){
    // use LK flow to estimate points in the right image

    // set the initial postion of features in right frame
    std::vector<cv::Point2f> vpointLeftKPs, vpointRightKPs;
    vpointLeftKPs.reserve(_mpCurrentFrame->mvpFeaturesLeft.size());
    vpointRightKPs.reserve(_mpCurrentFrame->mvpFeaturesLeft.size());
    for(auto &kp:  _mpCurrentFrame->mvpFeaturesLeft){
        vpointLeftKPs.push_back(kp->mkpPosition.pt);
        auto mp = kp->mpMapPoint.lock();
        if(mp){
            // use projected points as initial guess
            Vec2 px = _mpCameraRight->world2pixel(mp->Pos(), _mpCurrentFrame->Pose());
            vpointRightKPs.push_back(cv::Point2f(px[0], px[1]));
        }else{
            // use same pixel in left image
            vpointRightKPs.push_back(kp->mkpPosition.pt);
        }
    }

    // LK flow: calculate the keypoints' positions in right frame
    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK( _mpCurrentFrame->mLeftImg,
        _mpCurrentFrame->mRightImg, vpointLeftKPs, vpointRightKPs, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // create new feature objects in right frame
    int numGoodPoints = 0;
    _mpCurrentFrame->mvpFeaturesRight.reserve(status.size());
    for(size_t i = 0, N = status.size(); i < N; ++i){
        if(status[i]){
            cv::KeyPoint kp(vpointRightKPs[i], 7);
            Feature::Ptr feat(new Feature(kp));
            // Feature::Ptr feat(new Feature(_mpCurrentFrame, kp));
            feat->mbIsOnLeftFrame = false;
            _mpCurrentFrame->mvpFeaturesRight.push_back(feat);
            numGoodPoints++;
        } else{
            _mpCurrentFrame->mvpFeaturesRight.push_back(nullptr);
        }
    }

    LOG(INFO) << "Find " << numGoodPoints << " corresponding features in the right image.";
    return numGoodPoints;
}



// ----------------------------------------------------------------------------------------------

bool Frontend::BuildInitMap(){
    std::vector<SE3> poses{_mpCameraLeft->Pose(), _mpCameraRight->Pose()};
    size_t cntInitLandmarks = 0;
    for(size_t i = 0, N = _mpCurrentFrame->mvpFeaturesLeft.size(); i < N; i++){
        if(_mpCurrentFrame->mvpFeaturesRight[i] == nullptr)
            continue;
        // create mappoints by triangulation
        std::vector<Vec3> points{
            _mpCameraLeft->pixel2camera(
                Vec2(_mpCurrentFrame->mvpFeaturesLeft[i]->mkpPosition.pt.x,
                        _mpCurrentFrame->mvpFeaturesLeft[i]->mkpPosition.pt.y)),
            _mpCameraRight->pixel2camera(
                Vec2(_mpCurrentFrame->mvpFeaturesRight[i]->mkpPosition.pt.x,
                        _mpCurrentFrame->mvpFeaturesRight[i]->mkpPosition.pt.y))};
        Vec3 pworld = Vec3::Zero();

        if(triangulation(poses, points, pworld) && pworld[2] > 0){
            // if successfully triangulate, then create new mappoint and insert it to the map
            MapPoint::Ptr newMapPoint(new MapPoint);
            newMapPoint->SetPos(pworld);
            _mpCurrentFrame->mvpFeaturesLeft[i]->mpMapPoint = newMapPoint;
            _mpCurrentFrame->mvpFeaturesRight[i]->mpMapPoint = newMapPoint;
            _mpMap->InsertMapPoint(newMapPoint);

            cntInitLandmarks++;
        }
    }

    KeyFrame::Ptr newKF = KeyFrame::CreateKF(_mpCurrentFrame);

    _mpMap->InsertKeyFrame(newKF);

    if(_mpBackend){
        // LOG(INFO) << "backend udpate the map.";
        _mpBackend->UpdateMap();
    }
    
    LOG(INFO) << "Initial map created with " << cntInitLandmarks << " map points.";
    return true;
}


// --------------------------------------------------------------------------------------

bool Frontend::InsertKeyFrame(){
    DetectFeatures();
    FindFeaturesInRight();
    TriangulateNewPoints();

    KeyFrame::Ptr newKF = KeyFrame::CreateKF(_mpCurrentFrame);
    _mpMap->InsertKeyFrame(newKF);

    LOG(INFO) << "Set frame " << newKF->mnFrameId << " as keyframe " << newKF->mnKFId;

    if(_mpViewer){
        _mpViewer->UpdateMap();
    }
    if(_mpBackend){
        LOG(INFO) << "backend udpate the map.";
        _mpBackend->UpdateMap();
    }

    return true;
}


// --------------------------------------------------------------------------------------

int Frontend::TriangulateNewPoints(){
    std::vector<SE3> poses{_mpCameraLeft->Pose(), _mpCameraRight->Pose()};
    SE3 currentPoseTwc = _mpCurrentFrame->Pose().inverse();
    size_t cntTriangulatedPts = 0;
    for(size_t i = 0, N = _mpCurrentFrame->mvpFeaturesLeft.size(); i < N; i++){
        if(_mpCurrentFrame->mvpFeaturesRight[i] == nullptr
            || ! _mpCurrentFrame->mvpFeaturesLeft[i]->mpMapPoint.expired())
            continue;
        // create mappoints by triangulation
        std::vector<Vec3> points{
            _mpCameraLeft->pixel2camera(
                Vec2(_mpCurrentFrame->mvpFeaturesLeft[i]->mkpPosition.pt.x,
                        _mpCurrentFrame->mvpFeaturesLeft[i]->mkpPosition.pt.y)),
            _mpCameraRight->pixel2camera(
                Vec2(_mpCurrentFrame->mvpFeaturesRight[i]->mkpPosition.pt.x,
                        _mpCurrentFrame->mvpFeaturesRight[i]->mkpPosition.pt.y))};
        Vec3 pcamera = Vec3::Zero();

        if(triangulation(poses, points, pcamera) && pcamera[2] > 0){
            // if successfully triangulate, then create new mappoint and insert it to the map
            MapPoint::Ptr newMapPoint(new MapPoint);
            newMapPoint->SetPos(currentPoseTwc * pcamera);
            _mpCurrentFrame->mvpFeaturesLeft[i]->mpMapPoint = newMapPoint;
            _mpCurrentFrame->mvpFeaturesRight[i]->mpMapPoint = newMapPoint;
            // add these new mappoints to the map
            _mpMap->InsertMapPoint(newMapPoint);

            cntTriangulatedPts++;
        }
    }
    LOG(INFO) << " trangluate " << cntTriangulatedPts << " new mappoints while inserting new keyframe." ;
    return cntTriangulatedPts;
}





} // namespace myslam