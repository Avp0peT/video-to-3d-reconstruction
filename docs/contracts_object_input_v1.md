# object-input-v1：统一照片／视频准备

范围：普通照片目录或本地视频→有来源记录的≤1024边长RGB PNG→原有photo-io-v1。提供只读inspect及接纳已有成功视频结果。**不是照片／视频到CAD整链完成**；不启动SAM3、SfM、OpenMVS或CAD，也不替代A/B任务。

## 照片

目录只枚举一层，≤1024个条目、2–96张JPEG/PNG；不截取前96，不猜mask，不递归。非照片／子目录记录忽略原因；文件名NFC/大小写别名冲突和链接拒绝。每文件≤64MiB，总原始字节≤512MiB；解码前读取尺寸头，单帧2..24000000像素范围，原模式/位深/透明度等继续按T01解码器严格拒绝，不强制丢alpha或降位深。

先按原始EXIF一次转正，再保持比例缩至最长边1024，不放大。短边按整数half-up取整，缩后小于2拒绝；仅实际缩小时用Pillow LANCZOS。规范输出为RGB8 PNG，无EXIF二次方向。PNG保存到新的data/real/<id>/frames或outputs/delegation/T01A/fixtures/<id>/frames，再交给原T01。不修改T01的“不resize”契约：新入口显式生成其上游派生图片。

每帧保留原图绝对路径／文件字节hash、原尺寸／方向、方向后与准备后尺寸、raw_to_oriented、oriented_to_prepared、raw_to_prepared及轴比例[out,in]。连续像素中心映射为x'=(x+0.5)*sx-0.5，y同理；该矩阵记录几何约定，不声称每个输出像素来自单个原像素，也不是相机标定。缩放明确标lossy_resampling，颜色仍未标定。prepared file hash与尺寸前缀RGB像素hash分别保留。

新的T01 frame_id属于派生PNG，不能冒称原JPEG身份；原图身份由单独source binding及变换映射保留。所有T01行及其输入索引保留，重复图不静默去掉。少于2个不同解码画面或尺寸不同仅标SfM不兼容，不强行拉伸／裁剪；照片准备成功不等于视角覆盖足够。

## 视频及接纳

video子命令调用已有scripts/prepare_video.py，保持其有限文件／时长／帧数、真实PTS、颜色和旋转规则及资源边界。新包装不改变原实现。adopt-video仅接纳SUCCESS且严格整数exit0的已有video-photo-v1结果，保留原data_kind；synthetic不能通过改开关变real。源视频、两工具、两实现文件、timeline、选中PTS、导出图像与T01绑定复核。

逐视频行的photo_input_index/frame_id、image完整binding、尺寸与解码像素hash必须指向同一张T01派生输入，不能交换图片对但保留ID。timeline元数据／decoded_count／整数原始index和PTS／固定等时间选择重新核对；不重解码视频或重新观测FFmpeg行为，标BOUND_EXISTING_REPORT_NOT_DECODE_REPLAY。

## 结果与恢复边界

RESULT.json顶层object-input-v1；INPUTS_PREPARED/exit0仅输入准备成功。保留source_bindings、implementation、runtime、frames、photo_io（manifest/run/frames及全部派生PNG、canonical PNG绑定）、各原图→T01身份映射。SfM兼容状态单列：READY_FOR_SEPARATE_SFM、INSUFFICIENT_UNIQUE_IMAGES、REQUIRES_EQUAL_DIMENSIONS。其余segmentation/sfm/surface/CAD均NOT_RUN、coverage NOT_EVALUATED、scale unknown、length_unit null。

参数／输入错误2，后端错误4，照片软预算超限5；视频子入口的2/4/5保留。不自动重试、覆盖或续跑失败产物。每个输出和数据集均须新建，项目输出／数据路径的重解析祖先拒绝。照片处理中错误保留全部行，当前ERROR、后续NOT_RUN、已生成PREPARED；部分准备图不能被inspect接收为成功。

inspect只读，必须成功receipt、当前实现/runtime及原始文件均匹配。照片重新解码原图、计算方向／缩放、核对全部metadata与PNG hash；T01实现摘要、manifest、严格计数、frame_id、索引和逐像素输入／输出一致重新核对。视频按上述已有记录和文件校验。返回INPUT_BINDINGS_VALID；不启动下游或把历史失败转成功。该机制是文件与记录一致性验证，不是签名认证或对抗性溯源系统。

照片默认120秒共享软预算，逐帧／T01前后检查；T01在协调进程调用、有限解码不能随时抢占，**不是硬超时或RAM上限**。准备和T01产物合计1GiB软输出限，结束时再核对。只读照片重算同样120秒软截止。视频继承已有120秒子进程总截止与512MiB输出软限。模型／原生重建／CAD需要各自明确预算，不能借此入口放行。
