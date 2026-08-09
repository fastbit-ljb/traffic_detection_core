import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { Menu, X } from 'lucide-react';
import { gsap } from 'gsap';
import './PillNav.css';

export interface PillNavItem {
  id: string;
  label: string;
  ariaLabel?: string;
}

interface PillNavProps {
  items: PillNavItem[];
  activeId: string;
  onSelect: (id: string) => void;
  className?: string;
  ease?: string;
  baseColor?: string;
  pillColor?: string;
  hoveredPillTextColor?: string;
  pillTextColor?: string;
}

export function PillNav({
  items,
  activeId,
  onSelect,
  className = '',
  ease = 'power3.out',
  baseColor = '#9acfab',
  pillColor = '#ffffff',
  hoveredPillTextColor = '#ffffff',
  pillTextColor = '#4d805b',
}: PillNavProps) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const circleRefs = useRef<Array<HTMLSpanElement | null>>([]);
  const timelinesRef = useRef<Array<gsap.core.Timeline | undefined>>([]);
  const activeTweensRef = useRef<Array<gsap.core.Tween | undefined>>([]);
  const mobileMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const timelines = timelinesRef.current;
    const activeTweens = activeTweensRef.current;
    const layout = () => {
      circleRefs.current.forEach((circle, index) => {
        if (!circle?.parentElement) return;

        const pill = circle.parentElement;
        const { width, height } = pill.getBoundingClientRect();
        const radius = ((width * width) / 4 + height * height) / (2 * height);
        const diameter = Math.ceil(2 * radius) + 2;
        const delta = Math.ceil(radius - Math.sqrt(Math.max(0, radius * radius - (width * width) / 4))) + 1;
        const originY = diameter - delta;
        const label = pill.querySelector<HTMLElement>('.pill-label');
        const hoverLabel = pill.querySelector<HTMLElement>('.pill-label-hover');

        circle.style.width = `${diameter}px`;
        circle.style.height = `${diameter}px`;
        circle.style.bottom = `-${delta}px`;
        gsap.set(circle, { xPercent: -50, scale: 0, transformOrigin: `50% ${originY}px` });
        if (label) gsap.set(label, { y: 0 });
        if (hoverLabel) gsap.set(hoverLabel, { y: height + 12, opacity: 0 });

        timelinesRef.current[index]?.kill();
        const timeline = gsap.timeline({ paused: true });
        timeline.to(circle, { scale: 1.2, xPercent: -50, duration: 1, ease, overwrite: 'auto' }, 0);
        if (label) timeline.to(label, { y: -(height + 8), duration: 1, ease, overwrite: 'auto' }, 0);
        if (hoverLabel) timeline.to(hoverLabel, { y: 0, opacity: 1, duration: 1, ease, overwrite: 'auto' }, 0);
        timelinesRef.current[index] = timeline;
      });
    };

    layout();
    window.addEventListener('resize', layout);
    document.fonts?.ready.then(layout).catch(() => undefined);

    return () => {
      window.removeEventListener('resize', layout);
      timelines.forEach((timeline) => timeline?.kill());
      activeTweens.forEach((tween) => tween?.kill());
    };
  }, [ease, items]);

  useEffect(() => {
    const menu = mobileMenuRef.current;
    if (!menu) return;
    if (isMobileMenuOpen) {
      gsap.set(menu, { visibility: 'visible' });
      gsap.fromTo(menu, { opacity: 0, y: 8 }, { opacity: 1, y: 0, duration: 0.24, ease, overwrite: 'auto' });
      return;
    }
    gsap.to(menu, {
      opacity: 0,
      y: 8,
      duration: 0.18,
      ease,
      overwrite: 'auto',
      onComplete: () => gsap.set(menu, { visibility: 'hidden' }),
    });
  }, [ease, isMobileMenuOpen]);

  const animatePill = (index: number, entering: boolean) => {
    const timeline = timelinesRef.current[index];
    if (!timeline) return;
    activeTweensRef.current[index]?.kill();
    activeTweensRef.current[index] = timeline.tweenTo(entering ? timeline.duration() : 0, {
      duration: entering ? 0.3 : 0.2,
      ease,
      overwrite: 'auto',
    });
  };

  const selectItem = (id: string) => {
    onSelect(id);
    setIsMobileMenuOpen(false);
  };

  const cssVars = {
    '--base': baseColor,
    '--pill-bg': pillColor,
    '--hover-text': hoveredPillTextColor,
    '--pill-text': pillTextColor,
  } as CSSProperties;

  return <div className="pill-nav-container" style={cssVars}>
    <nav className={`pill-nav ${className}`} aria-label="检测导航">
      <div className="pill-nav-items desktop-only">
        {items.map((item, index) => <button
          key={item.id}
          type="button"
          className={`pill${activeId === item.id ? ' is-active' : ''}`}
          aria-label={item.ariaLabel ?? item.label}
          aria-pressed={activeId === item.id}
          onClick={() => selectItem(item.id)}
          onMouseEnter={() => animatePill(index, true)}
          onMouseLeave={() => animatePill(index, false)}
        >
          <span className="hover-circle" aria-hidden="true" ref={(element) => { circleRefs.current[index] = element; }} />
          <span className="label-stack"><span className="pill-label">{item.label}</span><span className="pill-label-hover" aria-hidden="true">{item.label}</span></span>
        </button>)}
      </div>
      <button className="mobile-menu-button mobile-only" type="button" aria-label={isMobileMenuOpen ? '关闭检测导航' : '打开检测导航'} aria-expanded={isMobileMenuOpen} onClick={() => setIsMobileMenuOpen((open) => !open)}>
        {isMobileMenuOpen ? <X size={18} aria-hidden="true" /> : <Menu size={19} aria-hidden="true" />}
      </button>
    </nav>
    <div className="mobile-menu-popover mobile-only" ref={mobileMenuRef}>
      {items.map((item) => <button key={item.id} type="button" className={`mobile-menu-link${activeId === item.id ? ' is-active' : ''}`} onClick={() => selectItem(item.id)}>{item.label}</button>)}
    </div>
  </div>;
}
