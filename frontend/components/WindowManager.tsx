"use client";
import React, { useState, useRef, useEffect } from "react";
import Window from "./Window";
import SankeyDiagram from "./SankeyDiagram";
import ScatterPlot from "./ScatterPlot";
import ControlPanel from "./ControlPanel";
import Thumbnail from "./Thumbnail";

interface WindowData {
  id: string;
  title: string;
  sankeyCsv?: string;
  scatterCsv?: string;
}

export default function WindowManager() {
  const [windows, setWindows] = useState<WindowData[]>([]);
  const [zoomFactor, setZoomFactor] = useState(1.0);
  const [layerCounts, setLayerCounts] = useState<Record<string, number>>({});
  const windowRefs = useRef<HTMLDivElement[]>([]);
  const scatterRefs = useRef<HTMLDivElement[]>([]);
  const sankeyRefs = useRef<HTMLDivElement[]>([]);
  useEffect(() => {
    const handler = (e: any) => {
      const { windowId, count } = e.detail;
      setLayerCounts(prev => ({
        ...prev,
        [windowId]: count
      }));
    };
    window.addEventListener("scatter-layer-count", handler);
    return () => window.removeEventListener("scatter-layer-count", handler);
  }, []);

  useEffect(() => {
    const handler = (e: any) => {
      const payload = e.detail;
      // Set sankeyCsv and scatterCsv to undefined before sending request
      setWindows(prev =>
        prev.map(w =>
          w.id === payload.id
            ? { ...w, sankeyCsv: undefined, scatterCsv: undefined }
            : w
        )
      );
      fetch("http://127.0.0.1:5001/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          windowId: payload.id,
          dataSize: payload.dataSize,
          featureCount: payload.featureCount,
          model: payload.model
        })
      })
        .then(res => res.json())
        .then(json => {
          console.log("Generate done:", json);
          setWindows(prev =>
            prev.map(w =>
              w.id === json.windowId
                ? {
                    ...w,
                    sankeyCsv: `/data/${json.windowId}/real_id_layer_transitions/real_id_layer_transitions.csv`,
                    scatterCsv: `/data/${json.windowId}/scatter_plot_data_by_layer/scatter_plot_data_by_layer`
                  }
                : w
            )
          );
        })
        .catch(err => console.error("Generate error:", err));
    };
    window.addEventListener("window-generate", handler);
    return () => window.removeEventListener("window-generate", handler);
  }, []);

  const syncScatterScroll = (index: number, scrollLeft: number) => {
    scatterRefs.current.forEach((ref, i) => {
      if (i !== index && ref) ref.scrollLeft = scrollLeft;
    });
  };

  const openWindow = () => {
    const id = Date.now().toString();
    setWindows((prev) => [...prev, { id, title: `Window ${prev.length + 1}` }]);
  };

  const closeWindow = (id: string) => {
    setWindows((prev) => prev.filter((w) => w.id !== id));
  };

  useEffect(() => {
    windowRefs.current = windowRefs.current.slice(0, windows.length);
    scatterRefs.current = scatterRefs.current.slice(0, windows.length);
    sankeyRefs.current = sankeyRefs.current.slice(0, windows.length);
  }, [windows.length]);

  const syncSankeyScroll = (left: number) => {
    sankeyRefs.current.forEach((ref) => {
      if (ref) ref.scrollLeft = left;
    });
  };
  useEffect(() => {
    const handler = (e: any) => {
      syncSankeyScroll(e.detail);
    };
    window.addEventListener("sankey-scroll", handler);
    return () => window.removeEventListener("sankey-scroll", handler);
  }, []);

  // Add global zoom with shift+wheel
  useEffect(() => {
    const handler = (e: WheelEvent) => {
      if (!e.shiftKey) return;
      e.preventDefault();

      // macOS touchpad: Shift + wheel ⇒ horizontal scroll (deltaX)
      const dx = e.deltaX;

      // dx < 0 → zoom in, dx > 0 → zoom out
      const delta = dx < 0 ? 0.1 : -0.1;

      setZoomFactor(prev =>
        Math.max(1, Math.min(5, prev + delta))
      );
    };

    window.addEventListener("wheel", handler, { passive: false });
    return () => window.removeEventListener("wheel", handler);
  }, []);

  // (Thumbnail logic moved to Thumbnail component)

  return (
    <div
      style={{ width: "100%" }}
    >
      <div style={{ width: "100%", boxSizing: "border-box" }}>
        <button
          className="bg-blue-600 text-white px-4 py-2 rounded mb-4"
          onClick={openWindow}
        >
          + New Window
        </button>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 1fr)",
            width: "100%",
            boxSizing: "border-box"
          }}
        >
          {windows.map((w, idx) => {
            const scatterData = Array.from({ length: 15 }, (_, i) => ({
              x: Math.random() * 100,
              y: Math.random() * 100,
              label: `${idx}-${i}`,
            }));

            return (
              <div style={{ width:"100%", maxWidth:"100%", boxSizing:"border-box" }} key={w.id}>
                <div
                  ref={(el) => {
                    if (el) windowRefs.current[idx] = el;
                  }}
                  style={{
                    width: "100%",
                    maxWidth: "100%",
                    boxSizing: "border-box",
                    overflow: "hidden"
                  }}
                >
                  <Window
                    id={w.id}
                    title={w.title}
                    onClose={closeWindow}
                    totalWindows={windows.length}
                    zoomFactor={zoomFactor}
                  >
                    {/* <ControlPanel
                      onChange={(params) => {
                        console.log("User selected:", params);
                      }}
                    /> */}
                    {w.scatterCsv ? (
                      <ScatterPlot
                        ref={(el) => {
                          if (el) scatterRefs.current[idx] = el;
                        }}
                        onScroll={(scrollLeft) => syncScatterScroll(idx, scrollLeft)}
                        clusterFilePath={w.scatterCsv}
                        zoomFactor={zoomFactor}
                      />
                    ) : (
                      <div style={{ opacity: 0.5, padding: "1rem", fontSize: "0.9rem" }}>
                        Waiting for backend to generate Scatter Plot…
                      </div>
                    )}
                    {w.sankeyCsv ? (
                      <div
                        ref={(el) => {
                          if (el) sankeyRefs.current[idx] = el;
                        }}
                      >
                        <SankeyDiagram sampleFile={w.sankeyCsv} zoomFactor={zoomFactor} />
                      </div>
                    ) : (
                      <div style={{ opacity: 0.5, padding: "1rem", fontSize: "0.9rem" }}>
                        Waiting for backend to generate Sankey…
                      </div>
                    )}
                    <Thumbnail
                      windowId={w.id}
                      scatterRef={scatterRefs.current[idx]}
                      layerCount={layerCounts[w.scatterCsv || ""] || 1}
                      zoomFactor={zoomFactor}
                    />
                  </Window>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}