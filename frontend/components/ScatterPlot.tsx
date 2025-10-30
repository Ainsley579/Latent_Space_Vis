"use client";
import * as d3 from "d3";
import React, { useEffect, useRef } from "react";

export default function ScatterPlot() {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const width = 400;
    const height = 300;

    const data = Array.from({ length: 20 }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
    }));

    svg.attr("width", width).attr("height", height);

    svg
      .append("g")
      .selectAll("circle")
      .data(data)
      .join("circle")
      .attr("cx", (d) => d.x)
      .attr("cy", (d) => d.y)
      .attr("r", 5)
      .attr("fill", "#ff7f0e")
      .attr("opacity", 0.8);
  }, []);

  return <svg ref={ref}></svg>;
}