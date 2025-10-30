"use client";
import React, { useState } from "react";
import Window from "./Window";

interface WindowData {
  id: string;
  title: string;
  content: React.ReactNode;
}

export default function WindowManager() {
  const [windows, setWindows] = useState<WindowData[]>([]);

  const openWindow = () => {
    const id = Date.now().toString();
    setWindows((prev) => [
      ...prev,
      {
        id,
        title: `Window ${prev.length + 1}`,
        content: <p>This is window #{prev.length + 1}</p>,
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