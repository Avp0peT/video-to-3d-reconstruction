# 重叠候选域准备

`scripts/prepare_overlapping_mvs_domains.py --subset-root <subset01> --plan <surface01/plan.json> --output <fresh>`

仅CPU读取现有稀疏点、互逆track、图像、mask与validity；不执行原生程序、不加载神经模型、不修改相机或原图。现有世界坐标点按固定三锚FPS分组，真实观测生成种子，4倍保守网格做边缘加权最短路。最大代价96、重叠8、边缘权重4，300秒截止、2线程。

输出原mask字节控制组、三组候选mask、每帧覆盖图、未知图、种子/身份账本、原生可用视图集合、输入前后绑定、命令及退出码。空间组不是解剖部位；没有种子或无法到达的区域保持未知。空mask可以作为准备记录，但不能宣称通过当前原生runner的非空要求。必须先验证mask消费再执行区域实验，所有target RGB与遮挡物继续保留。
