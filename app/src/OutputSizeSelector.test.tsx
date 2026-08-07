import { describe, it, expect, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import OutputSizeSelector from './OutputSizeSelector';

describe('OutputSizeSelector', () => {
  it('默认显示 1080x1920', () => {
    const onChange = vi.fn();
    render(<OutputSizeSelector value={{ width: 1080, height: 1920 }} onChange={onChange} />);
    expect(document.querySelector('[data-testid="out-size"]')).toBeTruthy();
  });

  it('选择 720x1280 触发 onChange', () => {
    const onChange = vi.fn();
    render(<OutputSizeSelector value={{ width: 1080, height: 1920 }} onChange={onChange} />);
    fireEvent.click(document.querySelector('[data-testid="preset-720"]')!);
    expect(onChange).toHaveBeenCalledWith({ width: 720, height: 1280 });
  });
});
