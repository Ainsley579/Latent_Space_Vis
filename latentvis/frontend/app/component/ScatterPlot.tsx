"use client";

import { useEffect, useRef } from "react";
import * as d3 from "d3";

type ScatterPoint = {
  real_id: string;
  x: number;
  y: number;
  cluster_id: number;
};

type ScatterPlotProps = {
  dataPath: string;
  activeCluster?: number | null;
  activeIds?: string[] | null;
  modelKey?: string;
  layerKey?: string;
  onPointSelect?: (sel: { model?: string; layer?: string; cluster: number; real_id: string }) => void;
};

export default function ScatterPlot({ dataPath, activeCluster, activeIds, modelKey, layerKey, onPointSelect }: ScatterPlotProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    container.innerHTML = "";

    const ro = new ResizeObserver(() => draw());
    ro.observe(container);

    let lastRows: ScatterPoint[] | null = null;

    function draw() {
      if (!container || !lastRows) return;
      container.innerHTML = "";
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);

      const margin = { top: 8, right: 8, bottom: 8, left: 8 };
      const innerW = Math.max(1, width - margin.left - margin.right);
      const innerH = Math.max(1, height - margin.top - margin.bottom);

      const svg = d3.select(container).append("svg").attr("width", width).attr("height", height);
      const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

      const xExtent = d3.extent(lastRows, (d) => d.x) as [number, number];
      const yExtent = d3.extent(lastRows, (d) => d.y) as [number, number];
      const x = d3.scaleLinear().domain(xExtent).range([0, innerW]).nice();
      const y = d3.scaleLinear().domain(yExtent).range([innerH, 0]).nice();

      const maxCluster = d3.max(lastRows, (d) => d.cluster_id) ?? 0;
      const color = d3.scaleOrdinal<number, string>(d3.schemeTableau10).domain(d3.range(maxCluster + 1));

      const activeIdSet = activeIds ? new Set(activeIds) : null;
      g.append("g")
        .selectAll("circle")
        .data(lastRows)
        .enter()
        .append("circle")
        .attr("cx", (d) => x(d.x))
        .attr("cy", (d) => y(d.y))
        .attr("r", 2.2)
        .attr("fill", (d) => color(Math.max(0, d.cluster_id)))
        .attr("fill-opacity", (d) => {
          if (activeIdSet) return activeIdSet.has(String(d.real_id)) ? 0.95 : 0.12;
          if (activeCluster == null) return 0.85;
          return d.cluster_id === activeCluster ? 0.95 : 0.12;
        })
        .attr("stroke", (d) => {
          if (activeIdSet) return activeIdSet.has(String(d.real_id)) ? "#222" : "none";
          if (activeCluster == null) return "none";
          return d.cluster_id === activeCluster ? "#222" : "none";
        })
        .attr("stroke-width", 0.6)
        .style("cursor", "pointer")
        .on("click", (event, d) => {
          event.stopPropagation();
          onPointSelect?.({
            model: modelKey,
            layer: layerKey,
            cluster: d.cluster_id,
            real_id: String(d.real_id),
          });
        });
    }

    fetch(dataPath)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch scatter data");
        return res.json();
      })
      .then((json: ScatterPoint[]) => {
        lastRows = json;
        draw();
      })
      .catch((err) => {
        console.error(err);
        if (container) container.innerHTML = "<div style='color:red'>Failed to load scatter data.</div>";
      });

    return () => {
      ro.disconnect();
    };
  }, [dataPath, activeCluster, activeIds, modelKey, layerKey, onPointSelect]);

  return <div ref={containerRef} className="w-full h-full" />;
}
