import { useEffect, useId, useRef, useState, type ReactNode } from 'react';

type GooeyDeviceItem = {
  id: string;
  label: string;
  icon: ReactNode;
  disabled?: boolean;
  title?: string;
};

type GooeyDeviceSwitchProps = {
  activeId: string;
  items: GooeyDeviceItem[];
  ariaLabel: string;
  onSelect: (id: string) => void;
};

const PARTICLE_COUNT = 15;
const ANIMATION_TIME = 520;
const TIME_VARIANCE = 180;
const PARTICLE_DISTANCES: [number, number] = [62, 9];

export function GooeyDeviceSwitch({ activeId, items, ariaLabel, onSelect }: GooeyDeviceSwitchProps) {
  const gooeyFilterId = `gooey-device-${useId().replace(/:/g, '')}`;
  const containerRef = useRef<HTMLDivElement>(null);
  const navRef = useRef<HTMLUListElement>(null);
  const filterRef = useRef<HTMLSpanElement>(null);
  const textRef = useRef<HTMLSpanElement>(null);
  const activeIndex = Math.max(0, items.findIndex((item) => item.id === activeId));
  const previousActiveIndex = useRef(activeIndex);
  const previousExternalIndex = useRef(activeIndex);
  const timersRef = useRef<number[]>([]);
  const makeParticlesRef = useRef<() => void>(() => undefined);
  const [visualActiveIndex, setVisualActiveIndex] = useState(activeIndex);

  const noise = (value = 1) => value / 2 - Math.random() * value;
  const getXY = (distance: number, pointIndex: number) => {
    const angle = ((360 + noise(8)) / PARTICLE_COUNT) * pointIndex * (Math.PI / 180);
    return [distance * Math.cos(angle), distance * Math.sin(angle)];
  };

  const clearParticles = () => {
    timersRef.current.forEach((timer) => window.clearTimeout(timer));
    timersRef.current = [];
    filterRef.current?.querySelectorAll('.gooey-device-particle').forEach((particle) => particle.remove());
  };

  const makeParticles = () => {
    const filter = filterRef.current;
    if (!filter) return;
    clearParticles();
    filter.classList.remove('active');
    filter.style.setProperty('--gooey-time', `${ANIMATION_TIME * 2 + TIME_VARIANCE}ms`);

    for (let index = 0; index < PARTICLE_COUNT; index += 1) {
      const duration = ANIMATION_TIME * 2 + noise(TIME_VARIANCE * 2);
      const start = getXY(PARTICLE_DISTANCES[0], PARTICLE_COUNT - index);
      const end = getXY(PARTICLE_DISTANCES[1] + noise(5), PARTICLE_COUNT - index);
      const rotate = noise(10);
      const timer = window.setTimeout(() => {
        const particle = document.createElement('span');
        const point = document.createElement('span');
        particle.className = 'gooey-device-particle';
        point.className = 'gooey-device-point';
        particle.style.setProperty('--start-x', `${start[0]}px`);
        particle.style.setProperty('--start-y', `${start[1]}px`);
        particle.style.setProperty('--end-x', `${end[0]}px`);
        particle.style.setProperty('--end-y', `${end[1]}px`);
        particle.style.setProperty('--gooey-time', `${duration}ms`);
        particle.style.setProperty('--gooey-scale', `${1 + noise(.2)}`);
        particle.style.setProperty('--gooey-rotate', `${(rotate > 0 ? rotate + 5 : rotate - 5) * 10}deg`);
        particle.appendChild(point);
        filter.appendChild(particle);
        requestAnimationFrame(() => filter.classList.add('active'));
        const cleanupTimer = window.setTimeout(() => particle.remove(), duration);
        timersRef.current.push(cleanupTimer);
      }, 30);
      timersRef.current.push(timer);
    }
  };
  makeParticlesRef.current = makeParticles;

  const updateEffectPosition = (element: HTMLLIElement) => {
    const container = containerRef.current;
    const filter = filterRef.current;
    const text = textRef.current;
    if (!container || !filter || !text) return;
    const containerRect = container.getBoundingClientRect();
    const itemRect = element.getBoundingClientRect();
    const styles = { left: `${itemRect.x - containerRect.x}px`, top: `${itemRect.y - containerRect.y}px`, width: `${itemRect.width}px`, height: `${itemRect.height}px` };
    Object.assign(filter.style, styles);
    Object.assign(text.style, styles);
    text.textContent = element.textContent;
  };

  const activateItem = (index: number, element: HTMLLIElement, notify = false) => {
    if (items[index]?.disabled || visualActiveIndex === index) return;
    setVisualActiveIndex(index);
    updateEffectPosition(element);
    textRef.current?.classList.remove('active');
    void textRef.current?.offsetWidth;
    textRef.current?.classList.add('active');
    if (notify) onSelect(items[index].id);
  };

  useEffect(() => {
    if (previousExternalIndex.current !== activeIndex) {
      previousExternalIndex.current = activeIndex;
      setVisualActiveIndex(activeIndex);
    }
  }, [activeIndex]);

  useEffect(() => {
    const activeItem = navRef.current?.querySelectorAll('li')[visualActiveIndex];
    if (!activeItem) return;
    updateEffectPosition(activeItem);
    textRef.current?.classList.add('active');
    if (previousActiveIndex.current !== visualActiveIndex) {
      previousActiveIndex.current = visualActiveIndex;
      makeParticlesRef.current();
    }
    const observer = new ResizeObserver(() => updateEffectPosition(activeItem));
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [visualActiveIndex]);

  useEffect(() => () => clearParticles(), []);

  return <div className="gooey-device-switch" ref={containerRef} role="group" aria-label={ariaLabel}>
    <svg className="gooey-device-svg-definitions" aria-hidden="true" focusable="false">
      <defs>
        <filter id={gooeyFilterId} x="-100%" y="-100%" width="300%" height="300%" colorInterpolationFilters="sRGB">
          <feGaussianBlur in="SourceGraphic" stdDeviation="7" result="blur" />
          <feColorMatrix in="blur" type="matrix" values="1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 22 -10" result="goo" />
          <feComposite in="SourceGraphic" in2="goo" operator="atop" />
        </filter>
      </defs>
    </svg>
    <nav aria-label={ariaLabel}><ul ref={navRef}>{items.map((item, index) => <li key={item.id} className={visualActiveIndex === index ? 'active' : ''}><button type="button" title={item.title} disabled={item.disabled} onClick={(event) => {
      const itemElement = event.currentTarget.closest('li');
      if (itemElement instanceof HTMLLIElement) activateItem(index, itemElement, true);
    }}>{item.icon}<span>{item.label}</span></button></li>)}</ul></nav>
    <span className="gooey-device-effect filter" ref={filterRef} aria-hidden="true" style={{ filter: `url(#${gooeyFilterId})` }} />
    <span className="gooey-device-effect text" ref={textRef} aria-hidden="true" />
  </div>;
}
