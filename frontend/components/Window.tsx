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
  const [dataSize, setDataSize] = useState(5000);
  const [featureCount, setFeatureCount] = useState(100);
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
      <div className="flex flex-col border-b pb-2 mb-2 gap-2">
        <div className="flex justify-between items-center">

          {/* Left side group */}
          <div className="flex flex-row items-center gap-6">

            <h3
              className="font-semibold cursor-pointer"
              onClick={() => setCollapsed(!collapsed)}
            >
              {collapsed ? "▶ " : "▼ "} {title}
            </h3>

            <div className="flex flex-col">
              <label className="text-xs opacity-70">
                Data Size: <span className="font-semibold">{dataSize}</span>
              </label>
              <input
                type="range"
                min="0"
                max="10000"
                value={dataSize}
                className="w-32"
                onChange={(e) => {
                  const value = Number(e.target.value);
                  setDataSize(value);
                  window.dispatchEvent(
                    new CustomEvent("window-data-size-change", {
                      detail: value
                    })
                  );
                }}
              />
            </div>

            <div className="flex flex-col">
              <label className="text-xs opacity-70">
                Feature Count: <span className="font-semibold">{featureCount}</span>
              </label>
              <input
                type="range"
                min="0"
                max="500"
                value={featureCount}
                className="w-32"
                onChange={(e) => {
                  const value = Number(e.target.value);
                  setFeatureCount(value);
                  window.dispatchEvent(
                    new CustomEvent("window-feature-count-change", {
                      detail: value
                    })
                  );
                }}
              />
            </div>

            <button
              className="px-2 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
              onClick={() => {
                window.dispatchEvent(new CustomEvent("window-generate", { detail: id }));
              }}
            >
              Generate
            </button>

            <button
              className="px-2 py-1 bg-green-500 text-white rounded hover:bg-green-600"
              onClick={() => {
                window.dispatchEvent(new CustomEvent("window-save", { detail: id }));
              }}
            >
              Save
            </button>
          </div>

          {/* Right side actions */}

          <button
            className="text-red-500 hover:text-red-700"
            onClick={() => onClose(id)}
          >
            ✕
          </button>

        </div>
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