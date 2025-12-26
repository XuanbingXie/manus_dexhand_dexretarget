# originflow_dexretarget
```bash
mamba create -n anytwist python=3.11
mamba activate mjwarp
pip install mujoco mujoco-mjx warp-lang

cd mjlab
pip install -e . --no-build-isolation
```

## retarget

从 Manus 手套接收数据并实时重定向到机器人手，使用 MuJoCo 可视化。

Args:
    robot_name: 机器人标识符（shadow）
    retargeting_type: 重定向类型（vector, position, dexpilot）
    hand_type: 手的类型（right, left）
    udp_port: UDP 端口号，默认 5006

```bash
conda activate anyskill
python udp/realtime_manus_retargeting_dexorigin.py \
  --robot-name shadow \
  --retargeting-type dexpilot \
  --hand-type right
```


```bash
python udp/realtime_manus_retargeting_gaia16.py  \
  --retargeting-type vector \
  --hand-type right
```
```bash
python udp/realtime_manus_retargeting_gaia16.py --use-manus-direct
```

dex-retageting复现：
```bash
cd dex-retargeting/example/vector_retargeting
python3 detect_from_video.py \
  --robot-name gaia16 \
  --video-path data/human_hand_video.mp4 \
  --retargeting-type vector \
  --hand-type right \
  --output-path data/gaia16_joints.pkl
```
```bash
python3 render_robot_hand.py \
  --pickle-path data/gaia16_joints.pkl \
  --output-video-path data/gaia16.mp4 \
  --headless
```
dex-retageting复现：
```bash
cd dex-retargeting/example/vector_retargeting
python3 detect_from_video.py \
  --robot-name linker \
  --video-path data/human_hand_video.mp4 \
  --retargeting-type dexpilot \
  --hand-type right \
  --output-path data/linker_joints.pkl
```

```bash
python3 render_robot_hand.py \
  --pickle-path data/linker_joints.pkl \
  --output-video-path data/linker.mp4 \
  --headless
```