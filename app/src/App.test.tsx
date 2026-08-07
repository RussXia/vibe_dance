import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from './App';

// jsdom 不实现 HTMLVideoElement 尺寸与 canvas 2d，mock 之（VideoPreview 初始化/绘制依赖）
Object.defineProperty(HTMLVideoElement.prototype, 'videoWidth', {
  configurable: true,
  get: () => 1080,
});
Object.defineProperty(HTMLVideoElement.prototype, 'videoHeight', {
  configurable: true,
  get: () => 1920,
});
const ctx2d = {
  scale: vi.fn(),
  clearRect: vi.fn(),
  fillRect: vi.fn(),
  strokeRect: vi.fn(),
  fillStyle: '',
  strokeStyle: '',
  lineWidth: 1,
};
Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
  configurable: true,
  value: vi.fn(() => ctx2d),
});

describe('App', () => {
  it('renders title and open button', () => {
    render(<App />);
    expect(screen.getByText('Vibe Dance Editor')).toBeTruthy();
    expect(screen.getByRole('button', { name: '打开视频' })).toBeTruthy();
  });
});

describe('App 导出流程', () => {
  const makeApi = (overrides: Record<string, unknown> = {}) => ({
    openVideo: vi.fn().mockResolvedValue({ path: '/tmp/a.mp4' }),
    saveVideo: vi.fn().mockResolvedValue({ path: '/tmp/out.mp4' }),
    showInFolder: vi.fn().mockResolvedValue({ ok: true }),
    submitTask: vi.fn().mockResolvedValue({ task_id: 'abc', status: 'QUEUED' }),
    getTask: vi.fn().mockResolvedValue({ task_id: 'abc', status: 'DONE', progress: 100, message: '' }),
    startEngine: vi.fn().mockResolvedValue({ ok: true }),
    ...overrides,
  });

  // 渲染 App → 打开视频 → 触发 loadedMetadata（jsdom 不会自动触发）让 box/videoSize 非空，
  // 从而 startTracking 守卫可通过。
  const renderReadyApp = async (api: Record<string, unknown>) => {
    (window as any).api = api;
    const { container } = render(<App />);
    fireEvent.click(screen.getByRole('button', { name: '打开视频' }));
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /开始跟踪/ })).toBeTruthy();
    });
    const video = container.querySelector('[data-testid="viewport-video"]') as HTMLVideoElement;
    fireEvent.loadedMetadata(video);
  };

  beforeEach(() => {
    (window as any).api = makeApi();
  });

  it('打开视频后显示开始跟踪按钮', async () => {
    render(<App />);
    fireEvent.click(screen.getByRole('button', { name: '打开视频' }));
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /开始跟踪/ })).toBeTruthy();
    });
  });

  it('轮询 DONE 时显示导出完成', async () => {
    const getTask = vi.fn().mockResolvedValue({ task_id: 'abc', status: 'DONE', progress: 100, message: '' });
    await renderReadyApp(makeApi({ getTask }));
    fireEvent.click(screen.getByRole('button', { name: /开始跟踪/ }));
    await waitFor(() => {
      expect(screen.getByText('✅ 导出完成')).toBeTruthy();
    });
  });

  it('轮询 FAILED 时显示错误信息', async () => {
    const getTask = vi.fn().mockResolvedValue({ task_id: 'abc', status: 'FAILED', progress: 50, message: 'boom' });
    await renderReadyApp(makeApi({ getTask }));
    fireEvent.click(screen.getByRole('button', { name: /开始跟踪/ }));
    await waitFor(() => {
      // 失败状态会同时渲染在红色错误提示和「状态:」两处，故用 getAllByText
      expect(screen.getAllByText(/boom/).length).toBeGreaterThan(0);
    });
  });

  it('轮询先 RUNNING 展示进度再转 DONE', async () => {
    const getTask = vi
      .fn()
      .mockResolvedValueOnce({ task_id: 'abc', status: 'RUNNING', progress: 42, message: '' })
      .mockResolvedValueOnce({ task_id: 'abc', status: 'DONE', progress: 100, message: '' });
    await renderReadyApp(makeApi({ getTask }));
    fireEvent.click(screen.getByRole('button', { name: /开始跟踪/ }));
    // 先展示 RUNNING 进度
    await waitFor(() => {
      expect(screen.getByText(/跟踪中… 42%/)).toBeTruthy();
    });
    // 再展示完成
    await waitFor(() => {
      expect(screen.getByText('✅ 导出完成')).toBeTruthy();
    });
  });
});
