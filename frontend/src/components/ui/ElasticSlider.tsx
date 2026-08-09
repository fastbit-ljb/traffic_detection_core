import { animate, motion, useMotionValue, useMotionValueEvent, useTransform } from 'framer-motion';
import { useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';

const MAX_OVERFLOW = 42;

type ElasticSliderProps = {
  value: number;
  min: number;
  max: number;
  step?: number;
  ariaLabel: string;
  onChange: (value: number) => void;
};

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);
const decay = (value: number) => (2 * (1 / (1 + Math.exp(-(value / MAX_OVERFLOW))) - 0.5)) * MAX_OVERFLOW;

export function ElasticSlider({ value, min, max, step = 1, ariaLabel, onChange }: ElasticSliderProps) {
  const sliderRef = useRef<HTMLDivElement>(null);
  const pointerX = useMotionValue(0);
  const overflow = useMotionValue(0);
  const thumbScale = useMotionValue(1);
  const [region, setRegion] = useState<'left' | 'middle' | 'right'>('middle');
  const [isHovered, setIsHovered] = useState(false);
  const percentage = ((value - min) / (max - min)) * 100;
  const trackScaleX = useTransform(overflow, (latest) => 1 + latest / 220);
  const trackScaleY = useTransform(overflow, [0, MAX_OVERFLOW], [1, 0.78]);

  useMotionValueEvent(pointerX, 'change', (latest) => {
    const slider = sliderRef.current;
    if (!slider) return;
    const { left, right } = slider.getBoundingClientRect();
    const nextRegion = latest < left ? 'left' : latest > right ? 'right' : 'middle';
    setRegion(nextRegion);
    overflow.set(nextRegion === 'left' ? decay(left - latest) : nextRegion === 'right' ? decay(latest - right) : 0);
  });

  const setValueFromPointer = (event: PointerEvent<HTMLDivElement>) => {
    const slider = sliderRef.current;
    if (!slider) return;
    const { left, width } = slider.getBoundingClientRect();
    const rawValue = min + ((event.clientX - left) / width) * (max - min);
    onChange(clamp(Math.round(rawValue / step) * step, min, max));
    pointerX.set(event.clientX);
  };

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    animate(thumbScale, 1.16, { duration: 0.15 });
    setValueFromPointer(event);
  };

  const releasePointer = () => {
    animate(overflow, 0, { type: 'spring', bounce: 0.5 });
    animate(thumbScale, 1, { type: 'spring', bounce: 0.35 });
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const nextValue = event.key === 'ArrowLeft' || event.key === 'ArrowDown' ? value - step
      : event.key === 'ArrowRight' || event.key === 'ArrowUp' ? value + step : null;
    if (nextValue === null) return;
    event.preventDefault();
    onChange(clamp(nextValue, min, max));
  };

  return <div className="elastic-slider">
    <div
      ref={sliderRef}
      className="elastic-slider-root"
      role="slider"
      tabIndex={0}
      aria-label={ariaLabel}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={value}
      onKeyDown={handleKeyDown}
      onPointerEnter={() => setIsHovered(true)}
      onPointerLeave={() => setIsHovered(false)}
      onPointerDown={handlePointerDown}
      onPointerMove={(event) => { if (event.buttons > 0) setValueFromPointer(event); }}
      onPointerUp={releasePointer}
      onPointerCancel={releasePointer}
      onLostPointerCapture={releasePointer}
    >
      <motion.div className="elastic-slider-hover-shell" animate={{ scaleY: isHovered ? 1.65 : 1 }} transition={{ type: 'spring', stiffness: 360, damping: 24, mass: 0.35 }}>
        <motion.div className="elastic-slider-track-motion" style={{ scaleX: trackScaleX, scaleY: trackScaleY, transformOrigin: region === 'left' ? 'right' : 'left' }}>
          <span className="elastic-slider-track"><motion.span className="elastic-slider-range" animate={{ width: `${percentage}%` }} transition={{ type: 'spring', stiffness: 420, damping: 30, mass: 0.35 }} /></span>
        </motion.div>
      </motion.div>
      <span className="elastic-slider-thumb-anchor" style={{ left: `${percentage}%` }}><motion.span className="elastic-slider-thumb" style={{ scale: thumbScale }} /></span>
    </div>
  </div>;
}
