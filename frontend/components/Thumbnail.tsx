

"use client";
import React, { useEffect, useRef, useState } from "react";

interface ThumbnailProps {
  windowId: string;
  scatterRef: HTMLDivElement | null;
  layerCount: number;
  zoomFactor: number;
}

export default function Thumbnail({ windowId, scatterRef, layerCount, zoomFactor }: ThumbnailProps) {
  const thumbRef = useRef<HTMLDivElement | null>(null);
  const [layers, setLayers] = useState<number>(layerCount);

  useEffect(() => {
    setLayers(layerCount);
  }, [layerCount]);

  useEffect(() => {
    const thumb = thumbRef.current;
    if (!thumb || !scatterRef) return;

    function updateViewport() {
      const viewport = thumb.querySelector(".thumb-viewport") as HTMLDivElement;
      if (!viewport) return;

      const thumbWidth = thumb.clientWidth;
      const contentWidth = scatterRef.scrollWidth;
      const viewWidth = scatterRef.clientWidth;

      if (contentWidth <= 0) return;

      const ratio = viewWidth / contentWidth;
      const leftRatio = scatterRef.scrollLeft / contentWidth;

      viewport.style.width = `${ratio * thumbWidth}px`;
      viewport.style.left = `${leftRatio * thumbWidth}px`;
    }

    const viewport = document.createElement("div");
    viewport.className = "thumb-viewport";
    viewport.style.position = "absolute";
    viewport.style.top = "0px";
    viewport.style.height = "100%";
    viewport.style.border = "2px solid rgba(0,0,0,0.8)";
    viewport.style.background = "rgba(0,0,0,0.05)";
    viewport.style.zIndex = "10";
    viewport.style.pointerEvents = "auto";
    thumb.appendChild(viewport);

    updateViewport();
    scatterRef.addEventListener("scroll", updateViewport);

    let isDragging = false;
    let startX = 0;

    viewport.addEventListener("mousedown", (e) => {
      isDragging = true;
      startX = e.clientX;
      e.preventDefault();
    });

    window.addEventListener("mouseup", () => {
      isDragging = false;
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDragging) return;

      const diff = e.clientX - startX;
      startX = e.clientX;

      const thumbWidth = thumb.clientWidth;
      const contentWidth = scatterRef.scrollWidth;

      const scrollDelta = diff * (contentWidth / thumbWidth);
      scatterRef.scrollLeft += scrollDelta;
    });

    return () => {
      scatterRef.removeEventListener("scroll", updateViewport);
    };
  }, [scatterRef, zoomFactor]);

  return (
    <div
      id={`thumbnail-${windowId}`}
      ref={thumbRef}
      style={{
        width: "100%",
        height: "40px",
        marginTop: "8px",
        position: "relative",
        overflow: "hidden",
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        border: "1px solid #ccc",
        boxSizing: "border-box",
        background: "#f8f8f8"
      }}
    >
      {Array.from({ length: layers }).map((_, i) => (
        <div
          key={i}
          style={{
            flex: 1,
            height: "100%",
            borderRight: "1px solid #ddd",
            boxSizing: "border-box"
          }}
        ></div>
      ))}
    </div>
  );
}