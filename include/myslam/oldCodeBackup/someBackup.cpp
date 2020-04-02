/**
 * This file is just for some old code backup.
 * Please ignore this file.
 */

bool LoopClosing::DetectLoop(){
    // std::unique_lock<std::mutex> lck(_mmutexDatabase);
    std::vector<float> vScores;
    float maxScore = 0;
    // int cntSuspected = 0;
    unsigned long bestId = 0;
    for(auto &db:  _mvDatabase){
        if(_mpCurrentKF->mnKFId - db.first < 20) break;
        float similarityScore = _mpDeepLCD->score(_mpCurrentKF->mpDescrVector, db.second->mpDescrVector);
        if(similarityScore > maxScore){
            maxScore = similarityScore;
            bestId = db.first;
        }
        // if(similarityScore > _similarityThres2){
        //     cntSuspected++;
        // }
        if(similarityScore > 0.85){
            vScores.push_back(similarityScore);
            LOG(INFO) << "find potential Candidate KF: current KF " 
            << _mpCurrentKF->mnFrameId << ", candiate KF " << _mvDatabase.at(db.first)->mnFrameId << ", "
            << _mvDatabase.at(db.first)->mnKFId
            << ", score: " << similarityScore;
        }
    }
    for (auto iter = vScores.begin(); iter != vScores.end(); iter++){
        if(*iter == maxScore){
            vScores.erase(iter);
            break;
        }
    }
    std::pair<float, float> meanAndVar = VectorMeanAndVariance(vScores);
    LOG(INFO) << "mean: " << meanAndVar.first << ", stdev: " << meanAndVar.second;
    if(maxScore > meanAndVar.first + 3 * meanAndVar.second){
        LOG(INFO) << "!!! find potential Candidate KF: current KF " 
            << _mpCurrentKF->mnFrameId << ", candiate KF " << _mvDatabase.at(bestId)->mnFrameId << ", "
            << _mvDatabase.at(bestId)->mnKFId
            << ", score: " << maxScore;
    
        ShowManyImages("Image", 2, _mpCurrentKF->mImageLeft, 
        _mvDatabase.at(bestId)->mImageLeft);
        cv::waitKey(1);
    }

    return false;
}




    // calculate the similarity score normalizing factor
    float factor;
    if(_mpLastKF){
        factor = _mpDeepLCD->score(_mpCurrentKF->mpDescrVector, _mpLastKF->mpDescrVector);
    }else{
        factor = 0.8;
    }
    _mpCurrentKF->mfSimilarityScoreNormFactor = factor;
    LOG(INFO) << "factor: " << 1.0 / factor;
    


// ------------------------------------------------------------------------------------------



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