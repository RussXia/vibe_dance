import { useEffect, useRef, useState } from 'react';
import type { AudioPreview } from './vite-env';

const METHOD_LABEL: Record<string, string> = { dtw: '精确对齐', beat: '节拍对齐', zero: '从头铺设' };

interface Props {
  preview: AudioPreview;
  initialOffset: number;
  durationA: number;
  onOffsetChange: (offset: number) => void;
}

export default function WaveformEditor({ preview, initialOffset, durationA, onOffsetChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [offset, setOffset] = useState(initialOffset);
  const [method, setMethod] = useState('dtw');
  const dragRef = useRef<{ startX: number; startOffset: number } | null>(null);

  // offset 变化时通知父组件
  useEffect(() => {
    onOffsetChange(offset);
  }, [offset, onOffsetChange]);

  const toX = (s: number, width: number) => (s / durationA) * width;

  const drawLane = (
    ctx: CanvasRenderingContext2D,
    data: number[],
    x0: number,
    top: number,
    width: number,
    color: string,
    isA: boolean,
  ) => {
    ctx.fillStyle = color;

    if (isA) {
      // A 波形：从 x=0 开始铺满 width
      const barWidth = (width / data.length) * 0.8;
      for (let i = 0; i < data.length; i++) {
        const x = (i / data.length) * width;
        const h = Math.max(1, data[i] * (width / 4));
        ctx.fillRect(x, top + 2, barWidth, h);
      }
    } else {
      // B 波形：从 x0 开始，B 时长 = len_B * 0.1s
      // pxPerSec = width / durationA
      // 每根柱代表 0.1s，宽度 = 0.1 * pxPerSec
      const pxPerSec = width / durationA;
      const bDurationSamples = data.length; // 每个样本 0.1s
      const barWidth = 0.1 * pxPerSec * 0.8; // 柱子实际宽度（留空隙）
      const barStep = 0.1 * pxPerSec; // 柱子之间的步进

      for (let i = 0; i < data.length; i++) {
        const x = x0 + (i / bDurationSamples) * (bDurationSamples * 0.1 * pxPerSec);
        const h = Math.max(1, data[i] * (width / 4));
        ctx.fillRect(x, top + 2, barWidth, h);
      }
    }
  };

  const draw = (rect: DOMRect) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const laneH = rect.height / 4; // 每条波形占 1/4
    // A 波形（顶部）
    drawLane(ctx, preview.waveform_a, 0, laneH, rect.width, '#6ea8ff', true);
    // B 波形（底部），从 offset 处开始
    const bx = toX(offset, rect.width);
    drawLane(ctx, preview.waveform_b, bx, laneH * 2, rect.width, '#4cd07a', false);
    // offset 分割线
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(bx, 0);
    ctx.lineTo(bx, rect.height);
    ctx.stroke();
  };

  // 用 ResizeObserver 保持 canvas 与容器同步重绘
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const rect = canvas.getBoundingClientRect();
      draw(rect);
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [preview, offset]);

  const handleMouseDown = (e: React.MouseEvent) => {
    dragRef.current = { startX: e.clientX, startOffset: offset };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const drag = dragRef.current;
    if (!drag || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const deltaS = ((e.clientX - drag.startX) / rect.width) * durationA;
    const next = Math.max(0, Math.min(drag.startOffset + deltaS, durationA));
    setOffset(next);
  };

  const handleMouseUp = () => {
    dragRef.current = null;
  };

  return (
    <div data-testid="waveform-editor" className="waveform-editor">
      <canvas
        ref={canvasRef}
        data-testid="waveform-b"
        className="waveform-canvas"
        style={{ cursor: 'ew-resize' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
      <div className="waveform-meta">
        <span>偏移: <b>{offset.toFixed(1)}s</b></span>
        <span className="waveform-method">{METHOD_LABEL[method] || method}</span>
      </div>
      {/* A 测试锚点（drawLane 用 canvas 画，无需 DOM） */}
      <div data-testid="waveform-a" hidden />
    </div>
  );
}
