"use client";
import React, { useState } from "react";
import Window from "./Window";
import SankeyDiagram from "./SankeyDiagram";
import ScatterPlot from "./ScatterPlot";

interface WindowData {
  id: string;
  title: string;
  content: React.ReactNode;
}

export default function WindowManager() {
  const [windows, setWindows] = useState<WindowData[]>([]);

  const openWindow = () => {
    const id = Date.now().toString();
    const windowIndex = windows.length + 1;

    // 根据窗口序号生成不同的假数据
    const sankeyData =
      windowIndex % 2 === 1
        ? {
            id: windowIndex,
            nodes: [
              { name: "A" },
              { name: "B" },
              { name: "C" },
            ],
            links: [
              { source: 0, target: 1, value: 8 + windowIndex },
              { source: 1, target: 2, value: 4 + windowIndex },
            ],
          }
        : {
            id: windowIndex,
            nodes: [
              { name: "X" },
              { name: "Y" },
              { name: "Z" },
              { name: "W" },
            ],
            links: [
              { source: 0, target: 2, value: 5 + windowIndex },
              { source: 1, target: 2, value: 3 + windowIndex },
              { source: 1, target: 3, value: 6 + windowIndex },
            ],
          };

    const scatterData =
      windowIndex % 2 === 1
        ? Array.from({ length: 15 }, (_, i) => ({
            x: Math.random() * 100,
            y: Math.random() * 100,
            label: `A${windowIndex}-${i}`,
          }))
        : Array.from({ length: 15 }, (_, i) => ({
            x: Math.random() * 50 + 50,
            y: Math.random() * 50,
            label: `B${windowIndex}-${i}`,
          }));

    const content = (
      <div style={{ display: "flex", flexDirection: "column", gap: "1rem", alignItems: "center" }}>
        <SankeyDiagram data={sankeyData} />
        <ScatterPlot data={scatterData} />
      </div>
    );

    setWindows((prev) => [
      ...prev,
      {
        id,
        title: `Window ${prev.length + 1}`,
        content,
      },
    ]);
  };

  const closeWindow = (id: string) => {
    setWindows((prev) => prev.filter((w) => w.id !== id));
  };

  return (
    <div className="p-4">
      <button
        className="bg-blue-600 text-white px-4 py-2 rounded mb-4"
        onClick={openWindow}
      >
        + New Window
      </button>

      <div className="flex flex-wrap gap-2">
        {windows.map((w) => (
          <Window key={w.id} id={w.id} title={w.title} onClose={closeWindow} totalWindows={windows.length}>
            {w.content}
          </Window>
        ))}
      </div>
    </div>
  );
}