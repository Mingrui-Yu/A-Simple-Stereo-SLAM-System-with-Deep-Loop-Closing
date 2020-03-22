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
    
