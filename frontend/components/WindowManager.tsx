"use client";
import React, { useState, useRef, useEffect } from "react";
import Window from "./Window";
import SankeyDiagram from "./SankeyDiagram";
import ScatterPlot from "./ScatterPlot";
import ControlPanel from "./ControlPanel";

interface WindowData {
  id: string;
  title: string;
}

export default function WindowManager() {
  const [windows, setWindows] = useState<WindowData[]>([]);
  const windowRefs = useRef<HTMLDivElement[]>([]);
  const scatterRefs = useRef<HTMLDivElement[]>([]);

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
    <div style={{ display: "flex", width: "100%" }}>
      <div style={{ width: "80%", paddingRight: "0.5rem", boxSizing: "border-box" }}>
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
                  >
                    {/* <ControlPanel
                      onChange={(params) => {
                        console.log("User selected:", params);
                      }}
                    /> */}
                    <ScatterPlot
                      ref={(el) => {
                        if (el) scatterRefs.current[idx] = el;
                      }}
                      onScroll={(scrollLeft) =>
                        syncScatterScroll(idx, scrollLeft)
                      }
                      clusterFilePath={idx % 2 === 0 ? "/data/scatter_plot_data_by_layer_1" : "/data/scatter_plot_data_by_layer_2"}
                    />
                    <SankeyDiagram
                      sampleFile={idx % 2 === 0 ? "/data/real_id_layer_transitions_1/real_id_layer_transitions.csv" : "/data/real_id_layer_transitions_2/real_id_layer_transitions.csv"}
                    />
                  </Window>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <div style={{ width: "20%", minWidth: "20%", maxWidth: "20%", height: "200vh", overflowY: "auto", background: "#f7f7f7", padding: "1rem", boxSizing: "border-box" }}>
        {/* Right side panel placeholder */}
        <h3 style={{ fontWeight: "bold", marginBottom: "1rem" }}>Right Panel</h3>
      </div>
    </div>
  );
}