# stereo_slam_system

## 后续需要改进的地方




Frontend.cpp 中 每次 triangulate 创建 new mappoints 时，能不能只创建近一些的点（左右匹配特征点x坐标差较大，或者点的depth较小），这些点 triangulate 的精度会高一些，就像 ORB-SLAM2 中对 far points 和 close points 的区分

解决：添加了 mappoint 在 frontend 中的删除机制




网络训练的时候 或许可以考虑 类似高博论文中 最小化帧间特征向量的距离






