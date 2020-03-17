# stereo_slam_system

## 后续需要改进的地方

ORB 特征提取需要添加 mask

mappoint 中的 observation 或许可以不用 list，而是使用 unordered_set，加快在 remove observation 时的查找速度

Frontend::FindFeaturesInRight() 中有个 7 是 right frame 的 keypoint 的 size，可以让它等于 left frame 中的对应 keypoint 的size（虽然好像无所谓）(还有 Frontend::TrackLastFrame() 中也是一样)

map.cpp 中 Map::RemoveOldKeyframe() 中 frameToRemove = _mumpAllKeyFrames.at(minKFId); 可以在 active keyframes 中搜索就可以了吧


Frontend.cpp 中 每次 triangulate 创建 new mappoints 时，能不能只创建近一些的点（左右匹配特征点x坐标差较大，或者点的depth较小），这些点 triangulate 的精度会高一些，就像 ORB-SLAM2 中对 far points 和 close points 的区分

Frontend::EstimateCurrentPose() 中，是否可以考虑删除 outlilers 对应的 mappoints ？
