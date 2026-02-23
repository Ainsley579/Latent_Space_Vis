"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type ZoomPanProps = {
  children: React.ReactNode;
  className?: string;
  id?: string;
  maxScale?: number;
  panPadding?: number;
  onTransform?: (t: { x: number; y: number; scale: number }) => void;
  externalTransform?: { x: number; y: number; scale: number };
};

type Transform = {
  x: number;
  y: number;
  scale: number;
};

function clamp(val: number, min: number, max: number) {
  return Math.min(max, Math.max(min, val));
}

export default function ZoomPan({
  children,
  className,
  id,
  maxScale = 6,
  panPadding = 0,
  onTransform,
  externalTransform,
}: ZoomPanProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);
  const spacerRef = useRef<HTMLDivElement | null>(null);
  const [transform, setTransform] = useState<Transform>({ x: 0, y: 0, scale: 1 });
  const transformRef = useRef<Transform>({ x: 0, y: 0, scale: 1 });
  const [contentSize, setContentSize] = useState({ width: 1, height: 1 });
  const [showZoomControls, setShowZoomControls] = useState(false);
  const minScaleRef = useRef(0.01);
  const didInitialFitRef = useRef(false);
  const userInteractedRef = useRef(false);
  const suppressTransformEmitRef = useRef(false);

  const transformStyle = useMemo(
    () => ({
      transform: `scale(${transform.scale})`,
      transformOrigin: "0 0",
    }),
    [transform.scale]
  );

  useEffect(() => {
    transformRef.current = transform;
    if (!suppressTransformEmitRef.current) {
      onTransform?.(transform);
    }
  }, [transform]);

  useEffect(() => {
    if (!externalTransform) return;
    const container = containerRef.current;
    if (!container) return;
    const curr = transformRef.current;
    const ds = Math.abs(curr.scale - externalTransform.scale);
    const dx = Math.abs(curr.x - externalTransform.x);
    const dy = Math.abs(curr.y - externalTransform.y);
    if (dx < 0.5 && dy < 0.5 && ds < 0.001) return;

    const nextScale = clamp(externalTransform.scale, minScaleRef.current, maxScale);
    suppressTransformEmitRef.current = true;
    setTransform({ x: externalTransform.x, y: externalTransform.y, scale: nextScale });
    window.requestAnimationFrame(() => {
      const maxLeft = Math.max(0, container.scrollWidth - container.clientWidth);
      const maxTop = Math.max(0, container.scrollHeight - container.clientHeight);
      const nextLeft = clamp(panPadding - externalTransform.x, 0, maxLeft);
      const nextTop = clamp(panPadding - externalTransform.y, 0, maxTop);
      container.scrollLeft = nextLeft;
      container.scrollTop = nextTop;
      window.requestAnimationFrame(() => {
        suppressTransformEmitRef.current = false;
      });
    });
  }, [externalTransform]);

  function recomputeMinScale() {
    const container = containerRef.current;
    const content = contentRef.current;
    if (!container || !content) return;
    const cRect = container.getBoundingClientRect();
    const baseW = content.offsetWidth || 1;
    const baseH = content.offsetHeight || 1;
    // Guard against transient "not yet rendered" sizes that can cause oversized initial fit.
    if (baseW < 2 || baseH < 2) return;
    setContentSize({ width: baseW, height: baseH });

    const scaleToFitWidth = cRect.width / baseW;
    // Keep the previous behavior: viewport-width fit can be >1 when the canvas is wider.
    const minScale = 0.01;
    const fitScale = Math.max(0.01, Math.min(maxScale, scaleToFitWidth));
    minScaleRef.current = minScale;

    const currentScale = transformRef.current.scale;
    const nextScale =
      !didInitialFitRef.current || !userInteractedRef.current
        ? fitScale
        : Math.max(currentScale, minScale);
    if (Math.abs(nextScale - currentScale) > 0.0001) {
      setTransform((prev) => ({ ...prev, scale: nextScale }));
    }

    if (!didInitialFitRef.current || !userInteractedRef.current) {
      didInitialFitRef.current = true;
      window.requestAnimationFrame(() => {
        container.scrollLeft = panPadding;
        container.scrollTop = panPadding;
      });
    }
  }

  function emitTransformFromScroll(scale: number) {
    const container = containerRef.current;
    if (!container) return;
    setTransform({
      x: panPadding - container.scrollLeft,
      y: panPadding - container.scrollTop,
      scale,
    });
  }

  useEffect(() => {
    recomputeMinScale();
    const container = containerRef.current;
    const content = contentRef.current;
    if (!container || !content) return;
    const onScroll = () => {
      if (suppressTransformEmitRef.current) return;
      emitTransformFromScroll(transformRef.current.scale);
    };
    container.addEventListener("scroll", onScroll, { passive: true });

    const ro = new ResizeObserver(() => recomputeMinScale());
    ro.observe(container);
    ro.observe(content);
    return () => {
      container.removeEventListener("scroll", onScroll as EventListener);
      ro.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function stepZoom(direction: 1 | -1) {
    userInteractedRef.current = true;
    const container = containerRef.current;
    if (!container) return;
    const current = transformRef.current.scale;
    const nextScale = clamp(current * (direction > 0 ? 1.2 : 1 / 1.2), minScaleRef.current, maxScale);
    if (Math.abs(nextScale - current) < 0.0001) return;

    const cx = container.clientWidth / 2;
    const cy = container.clientHeight / 2;
    const contentX = (cx + container.scrollLeft - panPadding) / current;
    const contentY = (cy + container.scrollTop - panPadding) / current;
    const nextLeft = panPadding + contentX * nextScale - cx;
    const nextTop = panPadding + contentY * nextScale - cy;

    setTransform((prev) => ({ ...prev, scale: nextScale }));
    window.requestAnimationFrame(() => {
      const maxLeft = Math.max(0, container.scrollWidth - container.clientWidth);
      const maxTop = Math.max(0, container.scrollHeight - container.clientHeight);
      container.scrollLeft = clamp(nextLeft, 0, maxLeft);
      container.scrollTop = clamp(nextTop, 0, maxTop);
      emitTransformFromScroll(nextScale);
    });
  }

  const zoomBtnBaseStyle: React.CSSProperties = {
    width: 34,
    height: 34,
    border: "none",
    background: "rgba(255,255,255,0.96)",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#111827",
    cursor: "pointer",
    transition: "background 0.12s ease, transform 0.12s ease",
  };

  return (
    <div
      id={id}
      className={className}
      style={{ position: "relative", width: "100%", height: "100%" }}
      onMouseEnter={() => setShowZoomControls(true)}
      onMouseLeave={() => setShowZoomControls(false)}
    >
      <div
        ref={containerRef}
        style={{
          overflow: "auto",
          position: "relative",
          width: "100%",
          height: "100%",
        }}
      >
        <div
          ref={spacerRef}
          style={{
            width: `${contentSize.width * transform.scale + panPadding * 2}px`,
            height: `${contentSize.height * transform.scale + panPadding * 2}px`,
            position: "relative",
          }}
        >
          <div
            ref={contentRef}
            style={{
              display: "inline-block",
              position: "absolute",
              left: `${panPadding}px`,
              top: `${panPadding}px`,
              ...transformStyle,
            }}
          >
            {children}
          </div>
        </div>
      </div>
      <div
        style={{
          position: "absolute",
          right: 8,
          bottom: 8,
          display: "flex",
          gap: 0,
          zIndex: 30,
          pointerEvents: showZoomControls ? "auto" : "none",
          opacity: showZoomControls ? 1 : 0,
          transition: "opacity 0.15s ease",
          border: "1px solid #d4d4d8",
          borderRadius: 8,
          overflow: "hidden",
          boxShadow: "0 2px 8px rgba(0,0,0,0.18)",
          background: "rgba(255,255,255,0.9)",
        }}
      >
        <button
          type="button"
          onClick={() => stepZoom(1)}
          title="Zoom in"
          aria-label="Zoom in"
          style={{ ...zoomBtnBaseStyle, borderRight: "1px solid #e4e4e7" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "#f4f4f5";
            e.currentTarget.style.transform = "translateY(-1px)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "rgba(255,255,255,0.96)";
            e.currentTarget.style.transform = "translateY(0)";
          }}
          onMouseDown={(e) => {
            e.currentTarget.style.transform = "translateY(0)";
          }}
          onMouseUp={(e) => {
            e.currentTarget.style.transform = "translateY(-1px)";
          }}
        >
          <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
            <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </button>
        <button
          type="button"
          onClick={() => stepZoom(-1)}
          title="Zoom out"
          aria-label="Zoom out"
          style={zoomBtnBaseStyle}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "#f4f4f5";
            e.currentTarget.style.transform = "translateY(-1px)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "rgba(255,255,255,0.96)";
            e.currentTarget.style.transform = "translateY(0)";
          }}
          onMouseDown={(e) => {
            e.currentTarget.style.transform = "translateY(0)";
          }}
          onMouseUp={(e) => {
            e.currentTarget.style.transform = "translateY(-1px)";
          }}
        >
          <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
            <path d="M3 8h10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </button>
      </div>
    </div>
  );
}
