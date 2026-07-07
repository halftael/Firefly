---
title: VLA Dataset Format
draft: false
tags:
  - AI/Embodded
created: 2026-07-07 19:43:12
updated: 2026-07-07 19:43:12
published: 2026-07-07 19:43:12
category: tech blog
---
## RLDS(Reinforcement Learning Datasets)

Google 提出的开源标准，以前主要是为了RL准备的，现在也成为了组织VLA
数据的逻辑标准。**从大到小的结构是: step->episode->dataset**
episode和dataset的类型都是`tf.data.Dataset`,而step是一个`dict`.
包含这一部的各个数据，具体格式如下：

```json
{
    'image': RGB图像(1080x1920x3),            # 主相机
    'image_left': RGB图像(1080x1920x3),       # 左相机
    'image_right': RGB图像(1080x1920x3),      # 右相机
    'depth': 深度图(400x640x1),               # 深度信息
    'joint_states': 1x177 的关节状态向量,      # 机器人关节
    'end_effector_poses': 1x14 的末端位姿,    # 末端执行器
    'language_instruction': "Pick up the can", # 语言指令
    # ... 还有相机内参、外参、触觉反馈等
}
```

## LeRobot v3.0

LeRobot提供了对RLDS的具体实现，其结构如下所示：

```bash
dataset/
├── meta/
│   ├── info.json          # 数据集元信息：特征名、类型、形状、FPS等
│   ├── stats.json         # 全局统计信息：mean/std/min/max，用于归一化
│   ├── tasks.jsonl        # 任务描述与ID的映射
│   └── episodes/          # Episode元数据（Parquet格式）
│       └── chunk-000/
│           └── file-000.parquet
├── data/                  # 表格数据：状态、动作、时间戳等（Parquet格式）
│   └── chunk-000/
│       └── file-000.parquet  # 一个文件包含多个Episode的数据
└── videos/                # 视频数据：多相机画面（MP4格式）
    └── camera/
        └── chunk-000/
            └── file-000.mp4   # 一个文件包含多个Episode的视频帧
```

可见主要是把数据的大块分为两类： 本体感知数据和动作都放到parquet里面， 图片数据整合成一个视频

> [!note] ✏️ 对比LLM
> 这里的一个Episode 相当于大语言模型的一个文档
> 现在的VLA需要我们后续对他做tokenize
