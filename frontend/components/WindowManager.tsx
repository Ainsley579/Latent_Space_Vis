"use client";
import React, { useState, useRef, useEffect } from "react";
import Window from "./Window";
import SankeyDiagram from "./SankeyDiagram";
import ScatterPlot from "./ScatterPlot";

interface WindowData {
  id: string;
  title: string;
}

export default function WindowManager() {
  const [windows, setWindows] = useState<WindowData[]>([]);
  const windowRefs = useRef<HTMLDivElement[]>([]);
  const scatterRefs = useRef<HTMLDivElement[]>([]);

  const syncVerticalScroll = (index: number, scrollTop: number) => {
    windowRefs.current.forEach((ref, i) => {
      if (i !== index && ref) ref.scrollTop = scrollTop;
    });
  };

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
  }, [windows.length]);

  return (
    <div className="p-4">
      <button
        className="bg-blue-600 text-white px-4 py-2 rounded mb-4"
        onClick={openWindow}
      >
        + New Window
      </button>

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            windows.length <= 1
              ? "repeat(1, 1fr)" 
              : "repeat(2, minmax(0, 1fr))",
          gap: "0.5rem",
        }}
      >
        {windows.map((w, idx) => {
          const scatterData = Array.from({ length: 15 }, (_, i) => ({
            x: Math.random() * 100,
            y: Math.random() * 100,
            label: `${idx}-${i}`,
          }));

          const windowHeight = windows.length <= 2 ? "100vh" : "50vh";

          return (
            <div
              key={w.id}
              ref={(el) => {
                if (el) windowRefs.current[idx] = el;
              }}
              onScroll={(e) =>
                syncVerticalScroll(
                  idx,
                  (e.target as HTMLDivElement).scrollTop
                )
              }
              style={{
                height: windowHeight,
                overflowY: "auto",
                transition: "height 0.3s ease",
              }}
            >
              <Window
                id={w.id}
                title={w.title}
                onClose={closeWindow}
                totalWindows={windows.length}
              >
                <ScatterPlot
                  ref={(el) => {
                    if (el) scatterRefs.current[idx] = el;
                  }}
                  onScroll={(scrollLeft) =>
                    syncScatterScroll(idx, scrollLeft)
                  }
                  data={scatterData}
                />
                <SankeyDiagram
                  clusterFile={idx % 2 === 0 ? "/data/real_id_component_transitions/real_id_component_transitions.csv" : "/data/cluster_transition_matrices_2.csv"}
                  sampleFile={idx % 2 === 0 ? "/data/real_id_layer_transitions/real_id_layer_transitions.csv" : "/data/real_id_layer_transitions/real_id_layer_transitions.csv"}
                />
              </Window>
            </div>
          );
        })}
      </div>
    </div>
  );
}