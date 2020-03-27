# stereo_slam_system

## 后续需要改进的地方

ORB 特征提取需要添加 mask

mappoint 中的 observation 或许可以不用 list，而是使用 unordered_set，加快在 remove observation 时的查找速度

Frontend::FindFeaturesInRight() 中有个 7 是 right frame 的 keypoint 的 size，可以让它等于 left frame 中的对应 keypoint 的size（虽然好像无所谓）(还有 Frontend::TrackLastFrame() 中也是一样)

map.cpp 中 Map::RemoveOldKeyframe() 中 frameToRemove = _mumpAllKeyFrames.at(minKFId); 可以在 active keyframes 中搜索就可以了吧


Frontend.cpp 中 每次 triangulate 创建 new mappoints 时，能不能只创建近一些的点（左右匹配特征点x坐标差较大，或者点的depth较小），这些点 triangulate 的精度会高一些，就像 ORB-SLAM2 中对 far points 和 close points 的区分

Frontend::EstimateCurrentPose() 中，是否可以考虑删除 outlilers 对应的 mappoints ？

优化，以 atomic 原子操作代替 mutex?

对于 tracking 过程中 判定为 outlier 的 feature，能不能直接去掉？

backend 中 optimize 中 
```
auto rk = new g2o::RobustKernelHuber();
rk->setDelta(chi2_th);
```
能否用于 frontend 中

仔细想想发现，backend的优化结果对 frontend 并没有起效果，，frontend 中 的tracking 并不会参考backend优化有的结果。可能的修改方法：frontend得到新的KF与上一KF的相对位姿，期间backend会对上一KF的位姿优化，新的KF可以根据上一KF的位姿加相对位姿进行矫正


网络训练的时候 或许可以考虑 类似高博论文中 最小化帧间特征向量的距离

输入网络的图片,在resize 之前先 blur 应该是有效的,这一点在训练网络的时候也可以考虑.目前blur 的 kernal size 还没有仔细设置

关于 mask 是否有效，还需进一步讨论

可以尝试解决 LK flow tracking 过程中 keypoints 的 octive 丢失的问题

optimization 中 能否用 unordered_map 代替 map