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
  const [transform, setTransform] = useState<Transform>({ x: 0, y: 0, scale: 1 });
  const transformRef = useRef<Transform>({ x: 0, y: 0, scale: 1 });
  const minScaleRef = useRef(1);
  const draggingRef = useRef(false);
  const dragStartRef = useRef<{ x: number; y: number; tx: number; ty: number }>({ x: 0, y: 0, tx: 0, ty: 0 });

  const transformStyle = useMemo(
    () => ({
      transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
      transformOrigin: "0 0",
    }),
    [transform]
  );

  useEffect(() => {
    transformRef.current = transform;
    onTransform?.(transform);
  }, [transform]);

  useEffect(() => {
    if (!externalTransform) return;
    const curr = transformRef.current;
    const dx = Math.abs(curr.x - externalTransform.x);
    const dy = Math.abs(curr.y - externalTransform.y);
    const ds = Math.abs(curr.scale - externalTransform.scale);
    if (dx < 0.5 && dy < 0.5 && ds < 0.001) return;
    setTransform(externalTransform);
  }, [externalTransform]);

  function clampTransform(next: Transform, containerW: number, containerH: number, contentW: number, contentH: number) {
    const sw = contentW * next.scale;
    const sh = contentH * next.scale;
    let minX = containerW - sw - panPadding;
    let maxX = panPadding;
    let minY = containerH - sh - panPadding;
    let maxY = panPadding;

    // If content fits, allow limited panning slack.
    if (sw <= containerW) {
      minX = -panPadding;
      maxX = panPadding;
    }
    if (sh <= containerH) {
      minY = -panPadding;
      maxY = panPadding;
    }

    return {
      ...next,
      x: clamp(next.x, minX, maxX),
      y: clamp(next.y, minY, maxY),
    };
  }

  function recomputeMinScale() {
    const container = containerRef.current;
    const content = contentRef.current;
    if (!container || !content) return;
    const cRect = container.getBoundingClientRect();
    const baseW = content.offsetWidth || 1;
    const baseH = content.offsetHeight || 1;
    if (baseW <= 0 || baseH <= 0) return;

    const scaleToFitWidth = cRect.width / baseW;
    const minScale = Math.max(0.1, Math.min(1, scaleToFitWidth));
    minScaleRef.current = minScale;

    const next = clampTransform(
      { ...transform, scale: minScale },
      cRect.width,
      cRect.height,
      baseW,
      baseH
    );
    setTransform(next);
  }

  function handleWheelEvent(e: WheelEvent) {
    e.preventDefault();
    e.stopPropagation();
    const container = containerRef.current;
    const content = contentRef.current;
    if (!container || !content) return;
    const cRect = container.getBoundingClientRect();
    const iRect = content.getBoundingClientRect();
    const minScale = minScaleRef.current;

    const delta = -e.deltaY;
    const zoomFactor = 1 + delta * 0.0025;
    const current = transformRef.current;
    const nextScale = clamp(current.scale * zoomFactor, minScale, maxScale);
    if (nextScale === current.scale) return;

    const cx = e.clientX - cRect.left;
    const cy = e.clientY - cRect.top;
    const scaleRatio = nextScale / current.scale;
    const nextX = cx - (cx - current.x) * scaleRatio;
    const nextY = cy - (cy - current.y) * scaleRatio;

    const next = clampTransform(
      { x: nextX, y: nextY, scale: nextScale },
      cRect.width,
      cRect.height,
      iRect.width,
      iRect.height
    );
    setTransform(next);
  }

  useEffect(() => {
    recomputeMinScale();
    const container = containerRef.current;
    const content = contentRef.current;
    if (!container || !content) return;

    const wheelListener = (evt: WheelEvent) => handleWheelEvent(evt);
    container.addEventListener("wheel", wheelListener, { passive: false });

    const ro = new ResizeObserver(() => recomputeMinScale());
    ro.observe(container);
    ro.observe(content);
    return () => {
      container.removeEventListener("wheel", wheelListener as EventListener);
      ro.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function onMouseDown(e: React.MouseEvent) {
    if (e.button !== 0) return;
    draggingRef.current = true;
    dragStartRef.current = { x: e.clientX, y: e.clientY, tx: transform.x, ty: transform.y };
  }

  function onMouseMove(e: React.MouseEvent) {
    if (!draggingRef.current) return;
    const container = containerRef.current;
    const content = contentRef.current;
    if (!container || !content) return;
    const cRect = container.getBoundingClientRect();
    const iRect = content.getBoundingClientRect();
    const dx = e.clientX - dragStartRef.current.x;
    const dy = e.clientY - dragStartRef.current.y;
    const next = clampTransform(
      { x: dragStartRef.current.tx + dx, y: dragStartRef.current.ty + dy, scale: transform.scale },
      cRect.width,
      cRect.height,
      iRect.width,
      iRect.height
    );
    setTransform(next);
  }

  function onMouseUp() {
    draggingRef.current = false;
  }

  return (
    <div
      ref={containerRef}
      id={id}
      className={className}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseUp}
      onMouseUp={onMouseUp}
      style={{
        overflow: "hidden",
        position: "relative",
        cursor: draggingRef.current ? "grabbing" : "grab",
        touchAction: "none",
      }}
    >
      <div ref={contentRef} style={{ display: "inline-block", ...transformStyle }}>
        {children}
      </div>
    </div>
  );
}
