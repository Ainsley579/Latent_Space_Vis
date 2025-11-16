"use client";
import React, { useState } from "react";
import SankeyDiagram from "./SankeyDiagram";
import ScatterPlot from "./ScatterPlot";

interface WindowProps {
  id: string;
  title: string;
  onClose: (id: string) => void;
  children: React.ReactNode;
  totalWindows?: number;
  clusterFile?: string;
  sampleFile?: string;
}

export default function Window({ id, title, onClose, children, totalWindows = 1, clusterFile, sampleFile }: WindowProps) {
  const [collapsed, setCollapsed] = useState(false);
  // const width =
  //   totalWindows <= 1
  //     ? "100%"
  //     : "calc(50% - 0.5rem)";
  const width = "100%";
  const height = "auto";


  return (
    <div
      className="bg-white rounded shadow-md p-3 transition-all duration-300 ease-in-out box-border"
      style={{
        width: "100%",
        maxWidth: "100%",
        height: "auto",
        overflowX: "hidden",
      }}
    >
      <div className="flex justify-between items-center border-b pb-1 mb-2">
        <h3 className="font-semibold cursor-pointer" onClick={() => setCollapsed(!collapsed)}>
          {collapsed ? "▶ " : "▼ "} {title}
        </h3>
        <button
          className="text-red-500 hover:text-red-700"
          onClick={() => onClose(id)}
        >
          ✕
        </button>
      </div>
      {!collapsed && (
        <div style={{ maxHeight: collapsed ? 0 : "2000px", overflow: "hidden", transition: "max-height 0.3s ease" }}>
          {React.isValidElement(children)
            ? React.cloneElement(children, { clusterFile, sampleFile })
            : children}
        </div>
      )}
    </div>
  );
}