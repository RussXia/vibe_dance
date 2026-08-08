import { useEffect, useRef, useState } from 'react';
import WaveformEditor from './WaveformEditor';
import type { AudioPreview, AudioTaskInfo } from './vite-env';

type Phase = 'idle' | 'aligning' | 'aligned' | 'rendering' | 'done' | 'failed';

export default function AudioSwap() {
  const [videoAPath, setVideoAPath] = useState<string | null>(null);
  const [audioBPath, setAudioBPath] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [progress, setProgress] = useState(0);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [info, setInfo] = useState<AudioTaskInfo | null>(null);
  const [offset, setOffset] = useState(0);
  const [outputPath, setOutputPath] = useState<string | null>(null);
  const [message, setMessage] = useState('');
  const audioBRef = useRef<HTMLAudioElement | null>(null);
  const videoARef = useRef<HTMLVideoElement | null>(null);
  const isMountedRef = useRef(true);

  // Cleanup audio on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      audioBRef.current?.pause();
      videoARef.current?.pause();
      audioBRef.current = null;
      videoARef.current = null;
    };
  }, []);

  const poll = async (id: string, stopStatus: string[]): Promise<AudioTaskInfo | null> => {
    // eslint-disable-next-line no-constant-condition
    while (true) {
      if (!isMountedRef.current) return null;
      const st = await window.api.getAudioTask(id);
      if (!isMountedRef.current) return st;
      setProgress(st.progress);
      setInfo(st);
      if (stopStatus.includes(st.status)) return st;
      await new Promise((r) => setTimeout(r, 500));
    }
  };

  const startAlign = async () => {
    if (!videoAPath || !audioBPath) return;
    setPhase('aligning');
    setProgress(0);
    setMessage('');
    try {
      const res = await window.api.submitAudioTask({
        video_a_path: videoAPath,
        audio_b_path: audioBPath,
        output_path: '',
        params: {},
      });
      setTaskId(res.task_id);
      const st = await poll(res.task_id, ['DONE', 'FAILED', 'CANCELLED']);
      if (!st) return;
      if (st.status === 'DONE') {
        setOffset(st.align_result?.offset_seconds ?? 0);
        setPhase('aligned');
      } else {
        setPhase('failed');
        setMessage(st.message || '对齐失败');
      }
    } catch (e) {
      setPhase('failed');
      setMessage('引擎未启动或连接失败');
    }
  };

  const stopPlayback = () => {
    audioBRef.current?.pause();
    videoARef.current?.pause();
  };

  const playPreview = () => {
    const p = info?.preview;
    if (!p) return;
    // 停止上次
    stopPlayback();
    // 素材A 视频（静音，纯画面）+ 原速 B 音轨
    // 成品只保留 B 音轨（去掉 A 现场声），预览与之保持一致：只播 B
    const video = videoARef.current;
    if (video) {
      video.currentTime = 0;
      void video.play().catch(() => {
        /* 视频加载失败不影响音频试听 */
      });
    }
    audioBRef.current = new Audio('file://' + p.audio_b_path);
    const b = audioBRef.current;
    // B 起点在 A 时间轴上的位置 offset：
    // - offset >= 0：B 从 0 开始，延迟 offset 秒启动（A 前段静音）
    // - offset < 0：B 起点在 A 之前，A 的 0 时刻 = B 的 |offset| 秒处
    //   → B 跳到 |offset| 秒处立即播放，与视频同步
    if (offset < 0) {
      b.currentTime = Math.min(-offset, p.duration_b);
      void b.play().catch(() => {});
    } else {
      b.currentTime = 0;
      const delayMs = Math.max(0, offset * 1000);
      setTimeout(() => b.play().catch(() => {}), delayMs);
    }
  };

  const download = async () => {
    if (!taskId) return;
    const save = await window.api.saveVideo(`vibe_audio_swap_${Date.now()}.mp4`);
    if (!save) return;
    setPhase('rendering');
    setProgress(0);
    try {
      await window.api.renderAudioTask(taskId, {
        offset_seconds: offset,
        output_path: save.path,
      });
      const st = await poll(taskId, ['DONE', 'FAILED', 'CANCELLED']);
      if (!st) return;
      if (st.status === 'DONE') {
        setOutputPath(save.path);
        setPhase('done');
      } else {
        setPhase('failed');
        setMessage(st.message || '导出失败');
      }
    } catch (e) {
      setPhase('failed');
      setMessage('导出失败');
    }
  };

  const renderResult = () => {
    if (phase === 'aligning' || phase === 'rendering') {
      return (
        <div className="status">
          <span>{phase === 'aligning' ? '对齐中' : '导出中'}… {Math.round(progress)}%</span>
          <div className="progress-wrap">
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${Math.min(progress, 100)}%` }} />
            </div>
          </div>
        </div>
      );
    }
    if (phase === 'aligned') {
      const p = info?.preview;
      if (!p) return null;
      return (
        <>
          {/* 预览：素材A 视频（静音）+ 替换后音频混音 */}
          <div className="audio-preview-stage">
            <video
              ref={videoARef}
              data-testid="audio-preview-video"
              src={p.video_a_path}
              muted
              loop
              playsInline
              style={{ width: '100%', display: 'block', background: '#000', borderRadius: 'var(--radius-sm)' }}
            />
          </div>
          <WaveformEditor
            preview={p}
            initialOffset={offset}
            durationA={p.duration_a}
            onOffsetChange={setOffset}
          />
          <div className="export-actions" style={{ marginTop: 12 }}>
            <button className="btn btn-secondary" onClick={playPreview}>
              播放
            </button>
            <button className="btn btn-secondary" onClick={stopPlayback}>
              停止
            </button>
            <button className="btn" onClick={download}>
              下载
            </button>
          </div>
        </>
      );
    }
    if (phase === 'done') {
      return (
        <div className="export-result">
          <div className="status ok">✅ 导出完成</div>
          {outputPath && (
            <>
              <span className="video-path">{outputPath}</span>
              <div className="export-actions">
                <button
                  className="btn btn-secondary"
                  onClick={() => window.api.showInFolder(outputPath)}
                >
                  打开所在文件夹
                </button>
              </div>
            </>
          )}
        </div>
      );
    }
    if (phase === 'failed') {
      return <div className="status err">{message || '处理失败'}</div>;
    }
    return null;
  };

  return (
    <div className="card audio-swap">
      <div className="card-title">替换音轨 · 手动对齐</div>
      <div className="audio-swap-row">
        <button
          className="btn btn-secondary"
          onClick={async () => {
            const r = await window.api.openVideo();
            if (r) {
              setVideoAPath(r.path);
              setPhase('idle');
            }
          }}
        >
          {videoAPath ? '更换素材A' : '选择素材A（视频）'}
        </button>
        <button
          className="btn btn-secondary"
          onClick={async () => {
            const r = await window.api.openAnyMedia();
            if (r) {
              setAudioBPath(r.path);
              setPhase('idle');
            }
          }}
        >
          {audioBPath ? '更换素材B' : '选择素材B（音乐/视频）'}
        </button>
      </div>
      {videoAPath && <p className="video-path">{videoAPath}</p>}
      {audioBPath && <p className="video-path">{audioBPath}</p>}
      <button
        className="btn btn-block"
        style={{ marginTop: 12 }}
        onClick={startAlign}
        disabled={!videoAPath || !audioBPath || phase === 'aligning' || phase === 'rendering'}
      >
        {phase === 'aligning' ? '处理中…' : '开始处理'}
      </button>
      <div style={{ marginTop: 12 }}>{renderResult()}</div>
    </div>
  );
}

