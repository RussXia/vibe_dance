import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import AudioSwap from './AudioSwap';

// Mock Audio class (jsdom doesn't implement Web Audio API)
class MockAudio {
  play() {}
  pause() {}
  src = '';
}
Object.defineProperty(window, 'Audio', { configurable: true, value: MockAudio });

// Mock ResizeObserver
class MockResizeObserver {
  constructor(callback: ResizeObserverCallback) {}
  observe() {}
  unobserve() {}
  disconnect() {}
}
Object.defineProperty(window, 'ResizeObserver', { configurable: true, value: MockResizeObserver });

// canvas + rect mock（WaveformEditor 依赖）
const ctx2d = {
  scale: vi.fn(),
  clearRect: vi.fn(),
  fillRect: vi.fn(),
  strokeRect: vi.fn(),
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  stroke: vi.fn(),
  fillStyle: '',
  strokeStyle: '',
  lineWidth: 1
};
Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
  configurable: true,
  value: vi.fn(() => ctx2d)
});
Object.defineProperty(HTMLElement.prototype, 'getBoundingClientRect', {
  configurable: true,
  value: vi.fn(() => ({
    left: 0,
    top: 0,
    width: 600,
    height: 120,
    right: 600,
    bottom: 120,
    x: 0,
    y: 0,
    toJSON: () => ({})
  }))
});

const api = {
  openVideo: vi.fn().mockResolvedValue({ path: '/tmp/a.mp4' }),
  openAudio: vi.fn().mockResolvedValue({ path: '/tmp/b.mp3' }),
  openAnyMedia: vi.fn().mockResolvedValue({ path: '/tmp/b.mp3' }),
  saveVideo: vi.fn().mockResolvedValue({ path: '/tmp/out.mp4' }),
  showInFolder: vi.fn().mockResolvedValue({ ok: true }),
  submitAudioTask: vi.fn().mockResolvedValue({ task_id: 'at1', status: 'QUEUED' }),
  getAudioTask: vi.fn().mockResolvedValue({
    task_id: 'at1',
    status: 'DONE',
    progress: 100,
    message: '',
    align_result: {
      offset_seconds: 2,
      tempo_ratio: 1,
      confidence: 'high',
      method: 'dtw'
    },
    preview: {
      video_a_path: '/tmp/a.mp4',
      audio_a_path: '/tmp/a.m4a',
      audio_b_path: '/tmp/b.m4a',
      waveform_a: [0.5, 0.6, 0.7],
      waveform_b: [0.4, 0.5, 0.6],
    },
  }),
  renderAudioTask: vi.fn().mockResolvedValue({ ok: true }),
};

beforeEach(() => {
  (window as any).api = api;
});

describe('AudioSwap', () => {
  it('mock API is available', () => {
    expect(window.api).toBeTruthy();
    expect(window.api.openVideo).toBeTruthy();
  });

  it('渲染素材选择按钮', () => {
    render(<AudioSwap />);
    expect(screen.getByRole('button', { name: /选择素材A/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /选择素材B/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /开始对齐/ })).toBeTruthy();
  });

  it('对齐完成后显示波形图与下载按钮', async () => {
    render(<AudioSwap />);
    const selectABtn = screen.getByRole('button', { name: /选择素材A/ });
    const selectBBtn = screen.getByRole('button', { name: /选择素材B/ });
    const alignBtn = screen.getByRole('button', { name: /开始对齐/ });

    // Use act() to wrap the async click
    await act(async () => {
      fireEvent.click(selectABtn);
      // Wait a bit for the promise to resolve and state to update
      await new Promise(r => setTimeout(r, 100));
    });

    // Now check if the path appears
    expect(screen.getByText('/tmp/a.mp4')).toBeTruthy();

    await act(async () => {
      fireEvent.click(selectBBtn);
      await new Promise(r => setTimeout(r, 100));
    });

    expect(screen.getByText('/tmp/b.mp3')).toBeTruthy();

    await act(async () => {
      fireEvent.click(alignBtn);
      // Need to wait longer for the polling to complete (poll is 500ms per iteration)
      // Mock returns immediately with DONE, so one iteration should be enough
      // But let's wait 2 seconds to be safe
      await new Promise(r => setTimeout(r, 2000));
    });

    // Wait for waveform
    expect(screen.getByTestId('waveform-editor')).toBeTruthy();
    expect(screen.getByRole('button', { name: /播放/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /下载/ })).toBeTruthy();
    // 视频预览：素材A 视频渲染在预览区
    const video = screen.getByTestId('audio-preview-video') as HTMLVideoElement;
    expect(video).toBeTruthy();
    expect(video.src).toContain('/tmp/a.mp4');
  });

  it('低置信度时提示手动微调', async () => {
    // Create a new mock API with low confidence
    const lowConfidenceApi = {
      openVideo: vi.fn().mockResolvedValue({ path: '/tmp/a.mp4' }),
      openAudio: vi.fn().mockResolvedValue({ path: '/tmp/b.mp3' }),
      openAnyMedia: vi.fn().mockResolvedValue({ path: '/tmp/b.mp3' }),
      saveVideo: vi.fn().mockResolvedValue({ path: '/tmp/out.mp4' }),
      showInFolder: vi.fn().mockResolvedValue({ ok: true }),
      submitAudioTask: vi.fn().mockResolvedValue({ task_id: 'at1', status: 'QUEUED' }),
      getAudioTask: vi.fn().mockResolvedValue({
        task_id: 'at1',
        status: 'DONE',
        progress: 100,
        message: '',
        align_result: {
          offset_seconds: 0,
          tempo_ratio: 1,
          confidence: 'low',
          method: 'beat'
        },
        preview: {
          video_a_path: '/tmp/a.mp4',
          audio_a_path: '/tmp/a.m4a',
          audio_b_path: '/tmp/b.m4a',
          waveform_a: [0.5],
          waveform_b: [0.4]
        },
      }),
      renderAudioTask: vi.fn().mockResolvedValue({ ok: true }),
    };
    (window as any).api = lowConfidenceApi;

    render(<AudioSwap />);

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /选择素材A/ }));
      await new Promise(r => setTimeout(r, 100));
    });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /选择素材B/ }));
      await new Promise(r => setTimeout(r, 100));
    });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /开始对齐/ }));
      await new Promise(r => setTimeout(r, 2000));
    });

    // Check if the hint text appears - the confidence hint div should contain it
    expect(screen.getByText(/对齐置信度较低/)).toBeTruthy();
  });

  it('下载流程：renderAudioTask 被调用', async () => {
    render(<AudioSwap />);

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /选择素材A/ }));
      await new Promise(r => setTimeout(r, 100));
    });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /选择素材B/ }));
      await new Promise(r => setTimeout(r, 100));
    });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /开始对齐/ }));
      await new Promise(r => setTimeout(r, 2000));
    });

    expect(screen.getByTestId('waveform-editor')).toBeTruthy();

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /下载/ }));
      await new Promise(r => setTimeout(r, 2000));
    });

    expect(api.saveVideo).toHaveBeenCalled();
    // 关键：renderAudioTask 必须收到用户选的 output_path（修复导出失败的根因）
    expect(api.renderAudioTask).toHaveBeenCalledWith('at1', {
      offset_seconds: 2,
      output_path: '/tmp/out.mp4',
    });
    expect(screen.getByRole('button', { name: /打开所在文件夹/ })).toBeTruthy();
  });
});
