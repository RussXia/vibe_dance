import { useState } from 'react';
import VideoPreview, { ViewportBox } from './VideoPreview';
import OutputSizeSelector from './OutputSizeSelector';
import AudioSwap from './AudioSwap';
import './App.css';

type TabKey = 'crop' | 'audio';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('crop');
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [status, setStatus] = useState('idle');
  const [box, setBox] = useState<ViewportBox | null>(null);
  const [videoSize, setVideoSize] = useState<{ width: number; height: number } | null>(null);
  const [outputSize, setOutputSize] = useState({ width: 1080, height: 1920 });
  const [progress, setProgress] = useState(0);
  const [outputPath, setOutputPath] = useState<string | null>(null);

  const openVideo = async () => {
    const res = await window.api.openVideo();
    if (res) {
      setVideoPath(res.path);
      setStatus('ready');
      setProgress(0);
    }
  };

  const handleBoxChange = (b: ViewportBox, size: { width: number; height: number }) => {
    setBox(b);
    setVideoSize(size);
  };

  const startTracking = async () => {
    if (!videoPath || !box || !videoSize) return;
    // 先让用户选择保存位置
    const defaultName = `vibe_dance_${Date.now()}.mp4`;
    const saveRes = await window.api.saveVideo(defaultName);
    if (!saveRes) return; // 用户取消保存对话框
    try {
      const res = await window.api.submitTask({
        video_path: videoPath,
        init_frame: 0,
        bbox: [box.x, box.y, box.w, box.h],
        output_size: [outputSize.width, outputSize.height],
        output_path: saveRes.path,
      });
      setOutputPath(saveRes.path);
      setStatus('tracking');
      setProgress(0);
      pollTask(res.task_id);
    } catch (e) {
      setStatus('failed: 引擎未启动或连接失败');
      setProgress(0);
    }
  };

  const pollTask = async (id: string) => {
    const st = await window.api.getTask(id);
    setProgress(st.progress);
    if (st.status === 'DONE') {
      setStatus('done');
      return;
    }
    if (st.status === 'FAILED') {
      setStatus(`failed: ${st.message}`);
      return;
    }
    setTimeout(() => pollTask(id), 500);
  };

  const renderStatus = () => {
    if (status === 'tracking') {
      return (
        <div className="status">
          <span>{progress < 70 ? '跟踪中' : '渲染中'}… {Math.round(progress)}%</span>
          <div className="progress-wrap">
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${Math.min(progress, 100)}%` }} />
            </div>
          </div>
        </div>
      );
    }
    if (status === 'done') {
      return (
        <div className="export-result">
          <div className="status ok">✅ 导出完成</div>
          {outputPath && (
            <>
              <span className="video-path">{outputPath}</span>
              <div className="export-actions">
                <button className="btn btn-secondary" onClick={() => window.api.showInFolder(outputPath)}>
                  打开所在文件夹
                </button>
              </div>
            </>
          )}
        </div>
      );
    }
    if (status.startsWith('failed')) {
      return <div className="status err">{status}</div>;
    }
    if (status === 'ready') {
      return <div className="status">已就绪 — 拖动取景框到目标人物，点「开始跟踪」导出</div>;
    }
    return <div className="status">打开一个视频开始</div>;
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-brand">
          <div className="app-logo">V</div>
          <div>
            <div className="app-title">Vibe Dance Editor</div>
            <div className="app-subtitle">视频自动跟随裁剪</div>
          </div>
        </div>
      </header>

      <nav className="app-tabs" role="tablist">
        <button
          role="tab"
          aria-selected={activeTab === 'crop'}
          data-testid="tab-crop"
          className={`app-tab${activeTab === 'crop' ? ' active' : ''}`}
          onClick={() => setActiveTab('crop')}
        >
          🎬 框选裁剪
        </button>
        <button
          role="tab"
          aria-selected={activeTab === 'audio'}
          data-testid="tab-audio"
          className={`app-tab${activeTab === 'audio' ? ' active' : ''}`}
          onClick={() => setActiveTab('audio')}
        >
          🎵 替换音轨
        </button>
      </nav>

      {activeTab === 'crop' ? (
        <div className="workspace">
        {/* 左侧：视频预览 + 取景框 */}
        <div className="card preview-card">
          <div className="card-title">预览 · 框选目标人物</div>
          {videoPath ? (
            <>
              <div className="preview-stage">
                <VideoPreview videoPath={videoPath} onBoxChange={handleBoxChange} />
              </div>
              <p className="video-path">{videoPath}</p>
            </>
          ) : (
            <div className="status-empty" style={{ padding: '40px 0', textAlign: 'center' }}>
              <div style={{ fontSize: 40, marginBottom: 12 }}>🎬</div>
              点击「打开视频」开始<br />支持 mp4 / mov
            </div>
          )}
        </div>

        {/* 右侧：控制面板 */}
        <div className="controls">
          <div className="card">
            <div className="card-title">视频</div>
            <button className="btn btn-block" onClick={openVideo}>
              {videoPath ? '更换视频' : '打开视频'}
            </button>
          </div>

          {videoPath && (
            <>
              <div className="card">
                <div className="card-title">输出尺寸</div>
                <OutputSizeSelector value={outputSize} onChange={setOutputSize} />
              </div>

              <div className="card">
                <div className="card-title">导出</div>
                <button
                  className="btn btn-block"
                  onClick={startTracking}
                  disabled={!box || status === 'tracking'}
                >
                  {status === 'tracking' ? '处理中…' : '开始跟踪并导出'}
                </button>
                <div style={{ marginTop: 12 }}>{renderStatus()}</div>
              </div>
            </>
          )}

          {!videoPath && (
            <div className="card">
              <div className="card-title">说明</div>
              <div className="status-empty">
                1. 打开一段多人视频<br />
                2. 用 9:16 取景框框住目标人物<br />
                3. 点击导出，自动跟踪并裁剪
              </div>
            </div>
          )}
        </div>
      </div>
      ) : (
        <div className="audio-page">
          <AudioSwap />
        </div>
      )}
    </div>
  );
}
