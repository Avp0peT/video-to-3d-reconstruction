# 从照片目录或视频准备重建输入

在项目根目录用现有项目Python运行`scripts/prepare_object.py`。这是统一输入前端，尚不会自动生成3D或CAD。

```powershell
# 整个照片目录：2–96张JPEG/PNG，子目录不扫描
python -B scripts/prepare_object.py photos --source 'D:\capture\object' --dataset-id object01 --data-kind real --output outputs/object01

# 一个本地视频：沿用真实时间戳抽帧入口
python -B scripts/prepare_object.py video --source 'D:\capture\object.mp4' --dataset-id object_video01 --data-kind real --output outputs/object_video01 --count 48 --ffmpeg '<绝对ffmpeg路径>' --ffprobe '<绝对ffprobe路径>'

# 复核当前输入，不重复运行视频或重建
python -B scripts/prepare_object.py inspect --run outputs/object01

# 已成功的视频准备结果可只读接纳，保留原synthetic/real类别
python -B scripts/prepare_object.py adopt-video --video-run '<原video-photo-v1目录>' --output outputs/adopted_video01
```

新照片会先应用EXIF，再按比例缩至最长边1024；原照片保持只读，输出记录说明是否有重采样。透明图、高位深、过大原图或损坏文件会明确失败。输出及数据集ID不能重复使用；失败保留记录，另起新运行并不自动获得重试额度。

`RESULT.json`中的photo_io.run.path是T01的run.json文件，**重建入口--io-run需要其父目录**。inspect会返回可直接定位的photo_io_run目录及sfm_compatibility。READY_FOR_SEPARATE_SFM仅表示至少2张不同画面且相同准备尺寸；不证明相机、视角覆盖或SfM一定成功。不同长宽比／方向导致不同尺寸时，请选择同一采集设置的数据，不自动拉伸。

真实工作下一步仍需按各独立任务完成SAM3、目标外形／跨视检查、SfM及显式分量选择、B去畸变mask身份链、原生MVS消费、表面和CAD验收。A pilot03和B任务既有执行额度不受本入口影响，本轮不代跑。

详细来源／限制见[契约](contracts_object_input_v1.md)。inspect会重新读照片并计算转换，可能耗时；视频接纳只复核保存的PTS、文件和记录，不重新解码，不应称为新的视频实测。
