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
    _numFeaturesInitGood = Config::Get<int>("numFeatures.initGood");
    _numFeaturesTrackingGood = Config::Get<int>("numFeatures.trackingGood");
    _numFeaturesTrackingBad = Config::Get<int>("numFeatures.trackingBad");

    int nFeaturesInit = Config::Get<int>("ORBextractor.nInitFeatures");
    float fScaleFactor = Config::Get<float>("ORBextractor.scaleFactor");
    int nLevels = Config::Get<int>("ORBextractor.nLevels");
    int fIniThFAST = Config::Get<int>("ORBextractor.iniThFAST");
    int fMinThFAST = Config::Get<int>("ORBextractor.minThFAST");

    _mbNeedUndistortion = Config::Get<int>("Camera.bNeedUndistortion");

    _mpORBextractorInit = ORBextractor::Ptr(new ORBextractor(nFeaturesInit,fScaleFactor,nLevels,fIniThFAST,fMinThFAST));
    // _mpORBextractor is created in system.cpp and introduced into frontend.cpp
}



// -------------------------------------------------------------------------------

bool Frontend::GrabStereoImage (const cv::Mat &leftImg, const cv::Mat &rightImg, 
    const double &dTimeStamp){

    _mpCurrentFrame = Frame::Ptr(new Frame(leftImg, rightImg, dTimeStamp));

    // undistort the images, which is not required in KITTI
    if(_mbNeedUndistortion){
        _mpCameraLeft->UndistortImage(_mpCurrentFrame->mLeftImg, _mpCurrentFrame->mLeftImg);
        _mpCameraRight->UndistortImage(_mpCurrentFrame->mRightImg, _mpCurrentFrame->mRightImg);
    }
    
    { // mutex: avoid conflict between frontend and loop correction
        std::unique_lock<std::mutex> lck(_mpMap->mmutexMapUpdate);

        switch(_mStatus){
            case FrontendStatus::INITING:
            StereoInit();
            break;
            case FrontendStatus::TRACKING_GOOD:
            case FrontendStatus::TRACKING_BAD:
                Track();
                break;
            case FrontendStatus::LOST:
                // Relocalization();
                // hasn't been implemented. (TO DO)
                return false; // now, if LOST, then just quit the system
                break;
        }
    } // mutex

    if(_mpViewer){
        _mpViewer->AddCurrentFrame(_mpCurrentFrame);
    }

    // usleep(1000 * 10);

    _mpLastFrame = _mpCurrentFrame;

    return true;
}


// -----------------------------------------------------------------------------

bool Frontend::Track(){

    // use constant velocity model to preliminarily estimiate the current frame's pose
    if(_mpLastFrame){
        _mpCurrentFrame->SetRelativePose(_mseRelativeMotion * _mpLastFrame->RelativePose());
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
        std::cout << "---------------------" << std::endl;
        std::cout << "Tracking LOST!" << std::endl;
        std::cout << "---------------------" << std::endl;
    }

    _mseRelativeMotion = _mpCurrentFrame->RelativePose() * _mpLastFrame->RelativePose().inverse();

    // detect new features / create new mappoints / create new KF
    if (_mStatus  == FrontendStatus::TRACKING_BAD){
        DetectFeatures();
        FindFeaturesInRight();
        TriangulateNewPoints();
        InsertKeyFrame();
    }

    return true;
}


// ------------------------------------------------------------------------------------------

int Frontend::TrackLastFrame(){
     // use LK flow to estimate points betweem last frame and current frame
    std::vector<cv::Point2f> pointKpsLast, pointKpsCurrent;
    pointKpsLast.reserve(_mpLastFrame->mvpFeaturesLeft.size());
    pointKpsCurrent.reserve(_mpLastFrame->mvpFeaturesLeft.size());
    for(auto &feat: _mpLastFrame->mvpFeaturesLeft){
        auto mp = feat->mpMapPoint.lock();
        if(mp && mp->mbIsOutlier == false){
            // if the feature links to a mappoint, use the reprojection result as the initial value
            auto px = _mpCameraLeft->world2pixel(mp->Pos(), _mpCurrentFrame->RelativePose() * _mpReferenceKF->Pose());
            pointKpsLast.push_back(feat->mkpPosition.pt);
            pointKpsCurrent.push_back(cv::Point2f(px[0], px[1]));
        } 
        else{
            // otherwise, use its position in last image as the initial value
            pointKpsLast.push_back(feat->mkpPosition.pt);
            pointKpsCurrent.push_back(feat->mkpPosition.pt);
        }
    }

    // LK flow estimation in opencv library
    std::vector<uchar> status;
    cv::Mat error;
    cv::calcOpticalFlowPyrLK(_mpLastFrame->mLeftImg, _mpCurrentFrame->mLeftImg,
        pointKpsLast, pointKpsCurrent, status, error, cv::Size(11, 11), 3, 
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    
    int numGoodPts = 0;
    _mpCurrentFrame->mvpFeaturesLeft.reserve(pointKpsCurrent.size());
    for(size_t i = 0, N = status.size(); i < N; i++){
        // if LK flow track successfully and the feature in last frame is linked to a mappoint
        // then, create a new feature in current frame
        if(status[i] && !_mpLastFrame->mvpFeaturesLeft[i]->mpMapPoint.expired()){
            cv::KeyPoint kp(pointKpsCurrent[i], 7); // only the position of keypoint is needed, so size 7 is just for creation with no meaning
            Feature::Ptr feature(new Feature(kp));
            feature->mpMapPoint = _mpLastFrame->mvpFeaturesLeft[i]->mpMapPoint;
            _mpCurrentFrame->mvpFeaturesLeft.push_back(feature);
            numGoodPts++;
        }
    }

    // LOG(INFO) << "Frontend: Find " << numGoodPts << " keypoints in the last frame.";
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

    // vertex
    VertexPose *vertex_pose = new VertexPose();
    vertex_pose->setId(0);
    vertex_pose->setEstimate(_mpCurrentFrame->RelativePose() * _mpReferenceKF->Pose());
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
        if(mp && mp->mbIsOutlier == false){
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
    int numIterations = 4;
    for(int iteration = 0; iteration < numIterations; iteration++){
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cntOutliers = 0;

        // count the outliers, outlier is not included in estimation until it is regarded as a inlier
        // somewhat like RANSAC
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

            // remove the robust kernel to see if it's outlier
            if(iteration == numIterations - 2){
                e->setRobustKernel(nullptr);
            }
        }
    }
   
    // set pose
    _mpCurrentFrame->SetPose(vertex_pose->estimate()); // this is only for viewer
    _mpCurrentFrame->SetRelativePose(
        vertex_pose->estimate() * _mpReferenceKF->Pose().inverse());

    // remove the link beween outlier features and their corresponding mappoints
    for(auto &feat: features){
        if(feat->mbIsOutlier){
            auto mp = feat->mpMapPoint.lock();

            // if the feature is regarded as a outlier after its mappoint just being created
            //      then set it as a outlier, which will be removed from the map later.
            if( mp && 
                    _mpCurrentFrame->mnFrameId - _mpReferenceKF->mnFrameId <= 2){
                mp->mbIsOutlier = true;
                _mpMap->AddOutlierMapPoint(mp->mnId);
            }

            feat->mpMapPoint.reset();
            feat->mbIsOutlier = false;
        }
    }

    //  LOG(INFO) << "Frontend: Outliers/Inliers in frontend current pose estimating: "  
        // << cntOutliers << "/" << features.size() - cntOutliers;

    return features.size() - cntOutliers;
}



// --------------------------------------------------------------

bool Frontend::StereoInit(){
    DetectFeatures();
    int numCoorFeatures = FindFeaturesInRight();
    if (numCoorFeatures < _numFeaturesInitGood){
        return false;
    }
    bool bBuildMapSuccess = BuildInitMap();

    if(bBuildMapSuccess){
        _mStatus = FrontendStatus::TRACKING_GOOD;
        return true;
    }
    return false;
}




// --------------------------------------------------------------

int Frontend::DetectFeatures(){

    // mask for ORB feature extraction
    cv::Mat mask(_mpCurrentFrame->mLeftImg.size(), CV_8UC1, 255);
    for (auto &feat: _mpCurrentFrame->mvpFeaturesLeft){
        cv::rectangle(mask, feat->mkpPosition.pt - cv::Point2f(20, 20),
            feat->mkpPosition.pt + cv::Point2f(20, 20), 0, CV_FILLED);
    }
    
    std::vector<cv::KeyPoint> keypoints;
    if(_mStatus == FrontendStatus::INITING){
        _mpORBextractorInit->Detect(_mpCurrentFrame->mLeftImg, mask, keypoints);
    } else{
        _mpORBextractor->Detect(_mpCurrentFrame->mLeftImg, mask, keypoints);
    }

    int cntDetected = 0;
    for(auto &kp: keypoints){
        Feature::Ptr newFeature(new Feature(kp));
        _mpCurrentFrame->mvpFeaturesLeft.push_back(newFeature);
        cntDetected++;
    }

    // LOG(INFO) << "Frontend: Detect " << cntDetected << " new features";

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
        if(mp && mp->mbIsOutlier == false){
            // use projected points as initial guess
            Vec2 px = _mpCameraRight->world2pixel(mp->Pos(), _mpCurrentFrame->RelativePose() * _mpReferenceKF->Pose());
            vpointRightKPs.push_back(cv::Point2f(px[0], px[1]));
        }else{
            // use same pixel position in left image
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
            cv::KeyPoint kp(vpointRightKPs[i], 7); // only the position of keypoint is needed, so size 7 is just for creation with no meaning
            Feature::Ptr feat(new Feature(kp));
            _mpCurrentFrame->mvpFeaturesRight.push_back(feat);
            numGoodPoints++;
        } else{
            _mpCurrentFrame->mvpFeaturesRight.push_back(nullptr);
        }
    }

    // LOG(INFO) << "Frontend: Find " << numGoodPoints << " corresponding features in the right image.";
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

    InsertKeyFrame();
    
    // LOG(INFO) << "Frontend: Initial map created with " << cntInitLandmarks << " map points.";
    return true;
}


// --------------------------------------------------------------------------------------

bool Frontend::InsertKeyFrame(){
    Vec6 se3_zero;
    se3_zero.setZero();

    KeyFrame::Ptr newKF = KeyFrame::CreateKF(_mpCurrentFrame);
    if(_mStatus == FrontendStatus::INITING){
        newKF->SetPose(Sophus::SE3d::exp(se3_zero));
    } else{
        newKF->SetPose(_mpCurrentFrame->RelativePose() * _mpReferenceKF->Pose());
        newKF->mpLastKF = _mpReferenceKF;
        newKF->mRelativePoseToLastKF = _mpCurrentFrame->RelativePose();
    }

    if(_mpBackend){
         _mpBackend->InsertKeyFrame(newKF);
    }

    // LOG(INFO) << "Frontend: Set frame " << newKF->mnFrameId << " as keyframe " << newKF->mnKFId;
    
    _mpReferenceKF = newKF;

    _mpCurrentFrame->SetRelativePose(Sophus::SE3d::exp(se3_zero));
    
    return true;
}


// --------------------------------------------------------------------------------------

int Frontend::TriangulateNewPoints(){
    std::vector<SE3> poses{_mpCameraLeft->Pose(), _mpCameraRight->Pose()};
    SE3 currentPoseTwc = (_mpCurrentFrame->RelativePose() * _mpReferenceKF->Pose()).inverse();
    size_t cntTriangulatedPts = 0;
    size_t cntPreviousMapPoints = 0;
    for(size_t i = 0, N = _mpCurrentFrame->mvpFeaturesLeft.size(); i < N; i++){
        if (!_mpCurrentFrame->mvpFeaturesLeft[i]->mpMapPoint.expired()){
            cntPreviousMapPoints++;
            continue;
        }
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
    // LOG(INFO) << "Frontend: Trangluate " << cntTriangulatedPts << " new mappoints, and now totally tracking " << cntTriangulatedPts + cntPreviousMapPoints << " mappoints."  ;
    return cntTriangulatedPts;
}





} // namespace myslam