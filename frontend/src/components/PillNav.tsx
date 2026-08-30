import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react';
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
  baseColor?: string;
  pillColor?: string;
  pillTextColor?: string;
  hoveredPillTextColor?: string;
  activeTextColor?: string;
}

const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Liquid segmented navigation: rude-mouse-79 look, heavy-dragonfly-92 sliding glider,
 * goo-filter droplet tail and elastic squash-and-stretch bounce. */
export function PillNav({
  items,
  activeId,
  onSelect,
  className = '',
  baseColor = '#eceef1',
  pillColor = '#ffffff',
  pillTextColor = '#5c6b7a',
  hoveredPillTextColor = '#111827',
  activeTextColor = '#111827',
}: PillNavProps) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const itemsRef = useRef<HTMLDivElement>(null);
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const labelRefs = useRef<Array<HTMLSpanElement | null>>([]);
  const gliderRef = useRef<HTMLSpanElement>(null);
  const tailRef = useRef<HTMLSpanElement>(null);
  const mobileMenuRef = useRef<HTMLDivElement>(null);
  const gliderStateRef = useRef({ x: 0, width: 0, ready: false });
  const activeIdRef = useRef(activeId);
  const firstActiveRenderRef = useRef(true);
  activeIdRef.current = activeId;

  const positionGlider = useCallback((targetId: string, animate: boolean) => {
    const glider = gliderRef.current;
    const tail = tailRef.current;
    if (!glider || !tail) return;
    const index = items.findIndex((item) => item.id === targetId);
    const tab = tabRefs.current[index];
    if (!tab) return;

    const x = tab.offsetLeft;
    const width = tab.offsetWidth;
    const centerX = x + width / 2;
    const previous = gliderStateRef.current;
    const shouldAnimate = animate && previous.ready && !prefersReducedMotion();

    if (!shouldAnimate || Math.abs(x - previous.x) < 2) {
      gsap.killTweensOf([glider, tail]);
      gsap.set(glider, { x, width, scaleX: 1, scaleY: 1 });
      gsap.set(tail, { x: centerX, scale: 1 });
      gliderStateRef.current = { x, width, ready: true };
      return;
    }

    const distance = Math.abs(x - previous.x);
    const stretch = Math.min(0.42, Math.max(0.1, (distance / Math.max(width, 1)) * 0.2));
    const previousCenter = previous.x + previous.width / 2;

    gsap.killTweensOf([glider, tail]);
    gsap.set(glider, { width });
    gsap.timeline()
      // 液滴拖尾慢半拍，被 goo 滤镜拉成液桥
      .fromTo(tail, { x: previousCenter, scale: 1 }, { x: centerX, duration: 0.7, ease: 'power2.in' }, 0)
      .fromTo(tail, { scale: 1.35 }, { scale: 1, duration: 0.65, ease: 'elastic.out(1, 0.45)' }, 0.05)
      // 主滑块冲过目标再弹回
      .to(glider, { x, duration: 0.72, ease: 'elastic.out(1, 0.6)' }, 0)
      // 移动中横向拉长、纵向压扁，落位后弹性恢复
      .fromTo(glider, { scaleX: 1 + stretch, scaleY: 1 - stretch * 0.6 }, { scaleX: 1, scaleY: 1, duration: 0.85, ease: 'elastic.out(1, 0.32)' }, 0.08);

    gliderStateRef.current = { x, width, ready: true };
  }, [items]);

  useEffect(() => {
    const layout = () => positionGlider(activeIdRef.current, false);
    layout();
    window.addEventListener('resize', layout);
    document.fonts?.ready.then(layout).catch(() => undefined);
    return () => window.removeEventListener('resize', layout);
  }, [positionGlider]);

  useEffect(() => {
    if (firstActiveRenderRef.current) {
      firstActiveRenderRef.current = false;
      return;
    }
    positionGlider(activeId, true);
  }, [activeId, positionGlider]);

  useEffect(() => {
    const menu = mobileMenuRef.current;
    if (!menu) return;
    if (isMobileMenuOpen) {
      gsap.set(menu, { visibility: 'visible' });
      gsap.fromTo(menu, { opacity: 0, y: 8 }, { opacity: 1, y: 0, duration: 0.24, ease: 'power3.out', overwrite: 'auto' });
      return;
    }
    gsap.to(menu, {
      opacity: 0,
      y: 8,
      duration: 0.18,
      ease: 'power3.out',
      overwrite: 'auto',
      onComplete: () => gsap.set(menu, { visibility: 'hidden' }),
    });
  }, [isMobileMenuOpen]);

  const squishLabel = (index: number, strong: boolean) => {
    const label = labelRefs.current[index];
    if (!label || prefersReducedMotion()) return;
    gsap.killTweensOf(label);
    gsap.timeline()
      .to(label, { scaleX: strong ? 1.12 : 1.07, scaleY: strong ? 0.84 : 0.9, duration: 0.14, ease: 'power2.out' })
      .to(label, { scaleX: 1, scaleY: 1, duration: 0.6, ease: 'elastic.out(1, 0.32)' });
  };

  const releaseLabel = (index: number) => {
    const label = labelRefs.current[index];
    if (!label || prefersReducedMotion()) return;
    gsap.killTweensOf(label);
    gsap.to(label, { scaleX: 1, scaleY: 1, duration: 0.35, ease: 'power2.out' });
  };

  const selectItem = (id: string, index: number) => {
    squishLabel(index, true);
    onSelect(id);
    setIsMobileMenuOpen(false);
  };

  const cssVars = {
    '--base': baseColor,
    '--pill-bg': pillColor,
    '--pill-text': pillTextColor,
    '--hover-text': hoveredPillTextColor,
    '--active-text': activeTextColor,
  } as CSSProperties;

  return <div className="pill-nav-container" style={cssVars}>
    <svg className="goo-defs" aria-hidden="true" focusable="false">
      <defs>
        <filter id="liquid-nav-goo" colorInterpolationFilters="sRGB">
          <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="blur" />
          <feColorMatrix in="blur" mode="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 24 -11" />
        </filter>
      </defs>
    </svg>
    <nav className={`liquid-nav ${className}`} aria-label="检测导航">
      <div className="liquid-nav-items desktop-only" ref={itemsRef}>
        <span className="liquid-layer" aria-hidden="true">
          <span className="liquid-glider" ref={gliderRef} />
          <span className="liquid-tail" ref={tailRef} />
        </span>
        {items.map((item, index) => <button
          key={item.id}
          type="button"
          ref={(element) => { tabRefs.current[index] = element; }}
          className={`liquid-tab${activeId === item.id ? ' is-active' : ''}`}
          aria-label={item.ariaLabel ?? item.label}
          aria-pressed={activeId === item.id}
          onClick={() => selectItem(item.id, index)}
          onMouseEnter={() => squishLabel(index, false)}
          onMouseLeave={() => releaseLabel(index)}
        >
          <span className="liquid-tab-label" ref={(element) => { labelRefs.current[index] = element; }}>{item.label}</span>
        </button>)}
      </div>
      <button className="mobile-menu-button mobile-only" type="button" aria-label={isMobileMenuOpen ? '关闭检测导航' : '打开检测导航'} aria-expanded={isMobileMenuOpen} onClick={() => setIsMobileMenuOpen((open) => !open)}>
        {isMobileMenuOpen ? <X size={18} aria-hidden="true" /> : <Menu size={19} aria-hidden="true" />}
      </button>
    </nav>
    <div className="mobile-menu-popover mobile-only" ref={mobileMenuRef}>
      {items.map((item) => <button key={item.id} type="button" className={`mobile-menu-link${activeId === item.id ? ' is-active' : ''}`} onClick={() => selectItem(item.id, items.indexOf(item))}>{item.label}</button>)}
    </div>
  </div>;
}
