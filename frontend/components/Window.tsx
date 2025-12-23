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
  hideControls?: boolean;
}

export default function Window({
  id,
  title,
  onClose,
  children,
  totalWindows = 1,
  clusterFile,
  sampleFile,
  hideControls = false,
}: WindowProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [dataSize, setDataSize] = useState(374);
  const [featureCount, setFeatureCount] = useState(128);
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
            {!hideControls && (
            <div className="flex flex-row items-center gap-6">
            <div className="flex flex-col">
              <label className="text-xs opacity-70">
                Data Size: <span className="font-semibold">{dataSize}</span>
              </label>
              <select
                value={dataSize}
                onChange={(e) => {
                  const value = Number(e.target.value);
                  setDataSize(value);
                  window.dispatchEvent(
                    new CustomEvent("window-data-size-change", { detail: value })
                  );
                }}
                className="border rounded px-2 py-1 text-sm"
              >
                <option value={374}>374</option>
                <option value={152}>152</option>
                <option value={3487}>3487</option>
              </select>
            </div>

            <div className="flex flex-col">
              <label className="text-xs opacity-70">
                Feature Count: <span className="font-semibold">{featureCount}</span>
              </label>
              <select
                value={featureCount}
                onChange={(e) => {
                  const value = Number(e.target.value);
                  setFeatureCount(value);
                  window.dispatchEvent(
                    new CustomEvent("window-feature-count-change", { detail: value })
                  );
                }}
                className="border rounded px-2 py-1 text-sm"
              >
                <option value={128}>128</option>
                <option value={256}>256</option>
                <option value={512}>512</option>
              </select>
            </div>

            <div className="flex flex-col">
              <label className="text-xs opacity-70">
                Model
              </label>
              <select
                id={`model-select-${id}`}
                onChange={(e) => {
                  window.dispatchEvent(
                    new CustomEvent("window-model-change", {
                      detail: e.target.value
                    })
                  );
                }}
                className="border rounded px-2 py-1 text-sm"
                defaultValue="noaug"
              >
                <option value="noaug">noaug</option>
                <option value="10conf">10conf</option>
                <option value="5conf">5conf</option>
                <option value="20conf">20conf</option>
                <option value="100conf">100conf</option>
              </select>
            </div>

            <button
              className="px-2 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
              onClick={() => {
                window.dispatchEvent(
                  new CustomEvent("window-generate", {
                    detail: {
                      id,
                      dataSize,
                      featureCount,
                      model: (document.querySelector(`#model-select-${id}`) as HTMLSelectElement)?.value || "noaug"
                    }
                  })
                );
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
            )}
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