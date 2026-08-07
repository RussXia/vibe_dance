import { useEffect, useRef, useState } from 'react';

const ASPECT = 9 / 16;
// 命中判定阈值（显示像素）：边/角区域宽度
const EDGE_PX = 16;

export interface ViewportBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

// 缩放锚点：拖哪个边/角，那个边/角固定
type EdgeKey = 'tl' | 'tr' | 'bl' | 'br' | 'left' | 'right' | 'top' | 'bottom';

interface Props {
  videoPath: string;
  onBoxChange: (box: ViewportBox, videoSize: { width: number; height: number }) => void;
}

export default function VideoPreview({ videoPath, onBoxChange }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [videoSize, setVideoSize] = useState<{ width: number; height: number }>({ width: 1080, height: 1920 });
  const [box, setBox] = useState<ViewportBox | null>(null);
  const [cursor, setCursor] = useState('move');
  const dragState = useRef<{
    mode: 'move' | 'resize';
    edge: EdgeKey | null;
    startX: number;
    startY: number;
    orig: ViewportBox;
  } | null>(null);

  // 视频元数据加载后，初始化居中的 9:16 取景框（占画面 80% 高度）
  const onLoadedMetadata = () => {
    const v = videoRef.current;
    if (!v) return;
    const vw = v.videoWidth;
    const vh = v.videoHeight;
    setVideoSize({ width: vw, height: vh });
    const h = Math.round(vh * 0.8);
    const w = Math.round(h * ASPECT);
    const x = Math.round((vw - w) / 2);
    const y = Math.round((vh - h) / 2);
    const initial = { x, y, w, h };
    setBox(initial);
    onBoxChange(initial, { width: vw, height: vh });
  };

  // 把取景框画到 canvas overlay 上
  useEffect(() => {
    const canvas = canvasRef.current;
    const v = videoRef.current;
    if (!canvas || !v || !box) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = v.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);
    // 换算到显示尺寸
    const scaleX = rect.width / videoSize.width;
    const scaleY = rect.height / videoSize.height;
    const bx = box.x * scaleX;
    const by = box.y * scaleY;
    const bw = box.w * scaleX;
    const bh = box.h * scaleY;
    // 框外暗化
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(0, 0, rect.width, rect.height);
    ctx.clearRect(bx, by, bw, bh);
    // 框线（白，稍亮）
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.strokeRect(bx, by, bw, bh);
    // 角 + 边中点把手（青色）
    ctx.fillStyle = '#4fc3f7';
    const handle = 7;
    // 四角
    ctx.fillRect(bx - handle / 2, by - handle / 2, handle, handle);
    ctx.fillRect(bx + bw - handle / 2, by - handle / 2, handle, handle);
    ctx.fillRect(bx - handle / 2, by + bh - handle / 2, handle, handle);
    ctx.fillRect(bx + bw - handle / 2, by + bh - handle / 2, handle, handle);
    // 四边中点
    const mh = 6;
    ctx.fillRect(bx + bw / 2 - mh / 2, by - mh / 2, mh, mh);
    ctx.fillRect(bx + bw / 2 - mh / 2, by + bh - mh / 2, mh, mh);
    ctx.fillRect(bx - mh / 2, by + bh / 2 - mh / 2, mh, mh);
    ctx.fillRect(bx + bw - mh / 2, by + bh / 2 - mh / 2, mh, mh);
  }, [box, videoSize]);

  // 事件换算：屏幕坐标 → 视频原始分辨率坐标
  const toVideoCoords = (clientX: number, clientY: number) => {
    const v = videoRef.current!;
    const rect = v.getBoundingClientRect();
    const px = (clientX - rect.left) / rect.width * videoSize.width;
    const py = (clientY - rect.top) / rect.height * videoSize.height;
    return { px, py };
  };

  // 判断鼠标命中的缩放锚点：返回边/角 key（显示坐标判定）
  const hitEdge = (clientX: number, clientY: number): EdgeKey | null => {
    const v = videoRef.current;
    const canvas = canvasRef.current;
    if (!v || !box || !canvas) return null;
    const rect = v.getBoundingClientRect();
    // 显示坐标下的框
    const scaleX = rect.width / videoSize.width;
    const scaleY = rect.height / videoSize.height;
    const bx = box.x * scaleX;
    const by = box.y * scaleY;
    const bw = box.w * scaleX;
    const bh = box.h * scaleY;
    const mx = clientX - rect.left;
    const my = clientY - rect.top;
    // 是否在框内
    if (mx < bx || mx > bx + bw || my < by || my > by + bh) return null;
    const nearL = mx < bx + EDGE_PX;
    const nearR = mx > bx + bw - EDGE_PX;
    const nearT = my < by + EDGE_PX;
    const nearB = my > by + bh - EDGE_PX;
    if (nearL && nearT) return 'tl';
    if (nearR && nearT) return 'tr';
    if (nearL && nearB) return 'bl';
    if (nearR && nearB) return 'br';
    if (nearL) return 'left';
    if (nearR) return 'right';
    if (nearT) return 'top';
    if (nearB) return 'bottom';
    return null;
  };

  // 根据命中锚点返回 CSS 光标
  const cursorFor = (edge: EdgeKey | null): string => {
    if (!edge) return 'move';
    switch (edge) {
      case 'tl': return 'nwse-resize';
      case 'br': return 'nwse-resize';
      case 'tr': return 'nesw-resize';
      case 'bl': return 'nesw-resize';
      case 'left': return 'ew-resize';
      case 'right': return 'ew-resize';
      case 'top': return 'ns-resize';
      case 'bottom': return 'ns-resize';
    }
  };

  // 未拖拽时移动鼠标：更新光标提示
  const handleHoverMove = (e: React.MouseEvent) => {
    if (dragState.current) return; // 拖拽中不更新
    const edge = hitEdge(e.clientX, e.clientY);
    const c = cursorFor(edge);
    if (c !== cursor) setCursor(c);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!box) return;
    const edge = hitEdge(e.clientX, e.clientY);
    const { px, py } = toVideoCoords(e.clientX, e.clientY);
    dragState.current = {
      mode: edge ? 'resize' : 'move',
      edge,
      startX: px,
      startY: py,
      orig: { ...box },
    };
    e.preventDefault();
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const drag = dragState.current;
    if (!drag || !box) return;
    const { px, py } = toVideoCoords(e.clientX, e.clientY);
    const dx = px - drag.startX;
    const dy = py - drag.startY;
    let next: ViewportBox;
    if (drag.mode === 'move') {
      next = {
        ...drag.orig,
        x: drag.orig.x + dx,
        y: drag.orig.y + dy,
      };
    } else {
      // 边/角缩放（图片缩放标准行为）：鼠标位置 = 被拖动的边/角，
      // 对边/对角锚定不动，保持 9:16。
      const orig = drag.orig;
      const edge = drag.edge;
      const right0 = orig.x + orig.w;
      const bottom0 = orig.y + orig.h;
      // 决定「被鼠标拖动的边」：鼠标位置决定这条边，对边锚定不动。
      // 角：鼠标所在的角，其两条邻边跟随鼠标，对边/对角锚定。
      let dragLeft = false, dragRight = false, dragTop = false, dragBottom = false;
      switch (edge) {
        case 'tl': dragLeft = true; dragTop = true; break;      // 左上角 → 左/上边跟随
        case 'tr': dragRight = true; dragTop = true; break;     // 右上角 → 右/上边跟随
        case 'bl': dragLeft = true; dragBottom = true; break;   // 左下角 → 左/下边跟随
        case 'br': dragRight = true; dragBottom = true; break;  // 右下角 → 右/下边跟随
        case 'left': dragLeft = true; break;
        case 'right': dragRight = true; break;
        case 'top': dragTop = true; break;
        case 'bottom': dragBottom = true; break;
      }
      // 新宽：被拖边=鼠标位置，对边锚定
      let newW = orig.w;
      let newH = orig.h;
      if (dragLeft) {
        newW = Math.max(orig.w * 0.1, right0 - px); // 左边界=鼠标x，右锚定
      } else if (dragRight) {
        newW = Math.max(orig.w * 0.1, px - orig.x); // 右边界=鼠标x，左锚定
      } else {
        const delta = Math.abs(dx) >= Math.abs(dy) ? dx : dy;
        newW = Math.max(orig.w * 0.1, orig.w + delta);
      }
      if (dragTop) {
        newH = Math.max(orig.h * 0.1, bottom0 - py); // 上边界=鼠标y，下锚定
      } else if (dragBottom) {
        newH = Math.max(orig.h * 0.1, py - orig.y); // 下边界=鼠标y，上锚定
      } else {
        const delta = Math.abs(dx) >= Math.abs(dy) ? dx : dy;
        newH = Math.max(orig.h * 0.1, orig.h + delta);
      }
      // 若拖的是角，用主轴方向的边来决定比例；否则独立缩放后按 9:16 归一
      // 统一：以拖动的水平边决定宽，再由 9:16 推高（拖上下边时以高推宽）
      const isCorner = edge === 'tl' || edge === 'tr' || edge === 'bl' || edge === 'br';
      const isHEdge = edge === 'left' || edge === 'right';
      const isVEdge = edge === 'top' || edge === 'bottom';
      if (isCorner || isHEdge) {
        newH = newW / ASPECT; // 宽决定高
      } else if (isVEdge) {
        newW = newH * ASPECT; // 高决定宽
      }
      // 锚定坐标：被拖边跟随鼠标位置，对边锚定
      let nx: number, ny: number;
      if (dragLeft) nx = px;                       // 左边界=鼠标x
      else if (dragRight) nx = right0 - newW;      // 右边界锚定，左边界=右-宽
      else nx = orig.x + (orig.w - newW) / 2;      // 仅上下边：水平居中
      if (dragTop) ny = py;                        // 上边界=鼠标y
      else if (dragBottom) ny = bottom0 - newH;    // 下边界锚定，上边界=下-高
      else ny = orig.y + (orig.h - newH) / 2;      // 仅左右边：垂直居中
      next = { x: Math.round(nx), y: Math.round(ny), w: Math.round(newW), h: Math.round(newH) };
    }
    // clamp 到画面内
    next.x = Math.max(0, Math.min(next.x, videoSize.width - next.w));
    next.y = Math.max(0, Math.min(next.y, videoSize.height - next.h));
    setBox(next);
    onBoxChange(next, videoSize);
  };

  const handleMouseUp = () => {
    dragState.current = null;
  };

  return (
    <div
      data-testid="viewport-container"
      onMouseMove={(e) => {
        handleHoverMove(e);
        handleMouseMove(e);
      }}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      style={{
        position: 'relative',
        width: '100%',
        maxWidth: 480,
        aspectRatio: videoSize.width > 0 && videoSize.height > 0
          ? `${videoSize.width} / ${videoSize.height}`
          : '9 / 16',
      }}
    >
      <video
        ref={videoRef}
        src={videoPath}
        onLoadedMetadata={onLoadedMetadata}
        data-testid="viewport-video"
        style={{ width: '100%', height: '100%', display: 'block', objectFit: 'contain', background: '#000' }}
      />
      <canvas
        ref={canvasRef}
        data-testid="viewport"
        onMouseDown={handleMouseDown}
        style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', cursor }}
      />
    </div>
  );
}
