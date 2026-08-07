# 验收清单

## 引擎（无 UI）

```bash
cd engine && source .venv/bin/activate
python -m pytest -v            # 全部通过
python ../scripts/e2e_smoke.py # 输出 SMOKE OK
```

## 桌面应用

```bash
cd app
npm test          # 全部通过
npm run electron:dev  # 手动验证：打开视频 → 框选 → 开始跟踪 → 进度 → 完成
```

注：本机 shell 设了 `ELECTRON_RUN_AS_NODE=1`，`npm run electron:dev` 已带 `env -u ELECTRON_RUN_AS_NODE` 保护，无需手动处理。

## 真实视频验收

1. 准备一段单机位固定拍摄、含目标人物的多人视频（mp4/mov）。
2. 打开应用，导入视频。
3. 用 9:16 取景框框住目标人物（拖动移动，拖边缘/角缩放）。
4. 点「开始跟踪」，选择保存位置，等待导出完成。
5. 打开输出视频，确认：目标人物全程在画面内、周围人被裁掉、画面不抖动、音轨保留。

## 已知限制（MVP）

- 取景框固定尺寸语义：人物走近可能出框、走远会被放大（可用 9:16 框选覆盖人物主要活动范围缓解）。
- 遮挡过久可能丢失，需增强重定位策略。
- 单机位固定拍摄（不支持运镜运动补偿）。
- 首次运行需加载 YOLO 模型（约 1-5s，含模型初始化；权重已随包分发避免联网下载）。
