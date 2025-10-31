"use client";
import * as d3 from "d3";
import React, { useEffect, useRef } from "react";

interface ScatterPlotProps {
  data?: { x: number; y: number; label?: string }[];
}

export default function ScatterPlot({ data }: ScatterPlotProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    container.innerHTML = "";

    const numPlots = Math.floor(Math.random() * 6) + 5; // 5~10 个散点图
    const height = 200;

    // 获取容器宽度（初始或默认 800）
    const containerWidth = container.clientWidth || 800;
    const svgWidth = containerWidth / 4; // ✅ 每个占容器宽度的 1/4

    for (let i = 0; i < numPlots; i++) {
      const svg = d3
        .select(container)
        .append("svg")
        .attr("height", height)
        .attr("width", svgWidth)
        .style("flex", "0 0 auto")
        .style("marginRight", "1rem")
        .style("border", "1px solid #ccc")
        .style("borderRadius", "8px")
        .style("background", "#fafafa");

      const localData =
        data ??
        Array.from({ length: 20 }, () => ({
          x: Math.random() * svgWidth,
          y: Math.random() * height,
        }));

      const color = d3.schemeCategory10[i % 10];

      svg
        .append("g")
        .selectAll("circle")
        .data(localData)
        .join("circle")
        .attr("cx", (d) => d.x)
        .attr("cy", (d) => d.y)
        .attr("r", 4)
        .attr("fill", color)
        .attr("opacity", 0.8);
    }
  }, [data]);

  return (
    <div
      ref={containerRef}
      style={{
        display: "flex",
        flexDirection: "row",
        overflowX: "auto",
        width: "100%",
        paddingBottom: "0.5rem",
      }}
    ></div>
  );
}