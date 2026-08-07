import { useState } from 'react';

const PRESETS = [
  { label: '720×1280', width: 720, height: 1280 },
  { label: '1080×1920', width: 1080, height: 1920 },
];

interface Props {
  value: { width: number; height: number };
  onChange: (v: { width: number; height: number }) => void;
}

export default function OutputSizeSelector({ value, onChange }: Props) {
  const [custom, setCustom] = useState(false);

  const isPreset = PRESETS.some((p) => p.width === value.width && p.height === value.height);

  const applyCustom = (wStr: string, hStr: string) => {
    const w = parseInt(wStr, 10);
    const h = parseInt(hStr, 10);
    if (Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0) {
      onChange({ width: w % 2 ? w + 1 : w, height: h % 2 ? h + 1 : h });
    }
  };

  return (
    <div data-testid="out-size">
      <div className="size-row">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            data-testid={`preset-${p.width}`}
            className={`size-preset${!custom && isPreset && value.width === p.width && value.height === p.height ? ' active' : ''}`}
            onClick={() => {
              setCustom(false);
              onChange({ width: p.width, height: p.height });
            }}
          >
            {p.label}
          </button>
        ))}
        <button
          data-testid="preset-custom"
          className={`size-preset${custom ? ' active' : ''}`}
          onClick={() => setCustom(true)}
        >
          自定义
        </button>
      </div>
      {custom && (
        <div className="size-custom">
          <input data-testid="custom-w" type="number" placeholder="宽" defaultValue={value.width}
            onChange={(e) => applyCustom(e.target.value, value.height.toString())} />
          <span>×</span>
          <input data-testid="custom-h" type="number" placeholder="高" defaultValue={value.height}
            onChange={(e) => applyCustom(value.width.toString(), e.target.value)} />
          <p className="size-hint">宽高需为偶数</p>
        </div>
      )}
    </div>
  );
}
