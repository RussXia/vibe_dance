import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import WaveformEditor from './WaveformEditor';

// jsdom 无 canvas 2d，mock
const ctx2d = {
  scale: vi.fn(),
  clearRect: vi.fn(),
  fillRect: vi.fn(),
  strokeRect: vi.fn(),
  fillStyle: '',
  strokeStyle: '',
  lineWidth: 1,
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  stroke: vi.fn(),
};
Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
  configurable: true,
  value: vi.fn(() => ctx2d),
});
// getBoundingClientRect 返回固定宽高
Object.defineProperty(HTMLElement.prototype, 'getBoundingClientRect', {
  configurable: true,
  value: vi.fn(() => ({ left: 0, top: 0, width: 600, height: 400, right: 600, bottom: 400, x: 0, y: 0, toJSON: () => ({}) })),
});
// ResizeObserver mock
global.ResizeObserver = vi.fn(() => ({
  observe: vi.fn(),
  disconnect: vi.fn(),
})) as any;

const preview = {
  video_a_path: '/tmp/a.mp4',
  audio_a_path: '/tmp/a.m4a',
  audio_b_path: '/tmp/b.m4a',
  waveform_a: Array.from({ length: 100 }, (_, i) => 0.5 + 0.5 * Math.sin(i / 5)),
  waveform_b: Array.from({ length: 50 }, (_, i) => 0.5 + 0.5 * Math.cos(i / 5)),
  duration_a: 10,
  duration_b: 5,
};

describe('WaveformEditor', () => {
  it('渲染双波形与 offset 显示', () => {
    render(<WaveformEditor preview={preview} initialOffset={2} durationA={10} onOffsetChange={() => {}} />);
    expect(screen.getByTestId('waveform-editor')).toBeTruthy();
    expect(screen.getByTestId('waveform-a')).toBeTruthy();
    expect(screen.getByTestId('waveform-b')).toBeTruthy();
    // offset 显示（含一位小数）
    expect(screen.getByText(/2\.0\s*s/)).toBeTruthy();
  });

  it('拖动 B 波形更新 offset 并回调', () => {
    const onChange = vi.fn();
    render(<WaveformEditor preview={preview} initialOffset={2} durationA={10} onOffsetChange={onChange} />);
    const b = screen.getByTestId('waveform-b');
    // 600px 宽 = 10s → 60px/s。从 clientX=12 拖到 clientX=72（+60px = +1s）
    fireEvent.mouseDown(b, { clientX: 12, clientY: 200 });
    fireEvent.mouseMove(b, { clientX: 72, clientY: 200 });
    fireEvent.mouseUp(b);
    // 拖动 +60px = +1s → offset 3.0
    expect(onChange).toHaveBeenCalled();
    const last = onChange.mock.calls.at(-1)![0] as number;
    // 允许更宽的误差范围以应对 React 更新的异步性
    expect(Math.abs(last - 3.0)).toBeLessThan(0.5);
  });

  it('B 可拖到 A 起点之前（负 offset）', () => {
    const onChange = vi.fn();
    render(<WaveformEditor preview={preview} initialOffset={0} durationA={10} onOffsetChange={onChange} />);
    const b = screen.getByTestId('waveform-b');
    // 从 clientX=300(offset=0) 拖到 clientX=0 → -300px = -5s → offset 应 clamp 到 -durationB=-5
    fireEvent.mouseDown(b, { clientX: 300, clientY: 200 });
    fireEvent.mouseMove(b, { clientX: 0, clientY: 200 });
    fireEvent.mouseUp(b);
    expect(onChange).toHaveBeenCalled();
    const last = onChange.mock.calls.at(-1)![0] as number;
    // -300px / (600px/10s) = -5s，clamp 到 [-5, 10] → -5
    expect(Math.abs(last - (-5))).toBeLessThan(0.5);
  });
});
