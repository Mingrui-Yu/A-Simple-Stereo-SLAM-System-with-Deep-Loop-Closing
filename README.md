# stereo_slam_system

## 后续需要改进的地方

ORB 特征提取需要添加 mask

mappoint 中的 observation 或许可以不用 list，而是使用 unordered_set，加快在 remove observation 时的查找速度

FrontEnd::FindFeaturesInRight() 中有个 7 是 right frame 的 keypoint 的 size，可以让它等于 left frame 中的对应 keypoint 的size（虽然好像无所谓）