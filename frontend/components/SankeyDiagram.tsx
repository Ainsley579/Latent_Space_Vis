"use client";
import * as d3 from "d3";
import {
  sankey,
  sankeyLinkHorizontal,
  SankeyNodeMinimal,
  SankeyLinkMinimal,
} from "d3-sankey";
import React, { useEffect, useRef, useState } from "react";

interface Node extends SankeyNodeMinimal<Node, Link> {
  name: string;
}

interface Link extends SankeyLinkMinimal<Node, Link> {
  value: number;
}

interface SankeyData {
  nodes: Node[];
  links: Link[];
}

export default function SankeyDiagram() {
  const ref = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [data, setData] = useState<SankeyData | null>(null);


  useEffect(() => {
    d3.csv("/data/sample_component_paths.csv").then(rows => {
      const layerNames = Object.keys(rows[0]).filter(k => k !== "sample_id");
      const links: Link[] = [];
      const nodeNames = new Set<string>();

      for (let i = 0; i < rows.length; i++) {
        for (let j = 0; j < layerNames.length - 1; j++) {
          const fromLayer = layerNames[j];
          const toLayer = layerNames[j + 1];
          const fromNode = `${fromLayer}_${rows[i][fromLayer]}`;
          const toNode = `${toLayer}_${rows[i][toLayer]}`;
          nodeNames.add(fromNode);
          nodeNames.add(toNode);

          const key = `${fromNode}->${toNode}`;
          const existing = links.find((l: any) => l.key === key);
          if (existing) existing.value += 1;
          else links.push({ source: fromNode, target: toNode, value: 1, key });
        }
      }

      const nodes = Array.from(nodeNames).map((name) => ({ name }));
      const nameToIndex = new Map(nodes.map((n, i) => [n.name, i]));
      links.forEach((l: any) => {
        l.source = nameToIndex.get(l.source);
        l.target = nameToIndex.get(l.target);
      });

      setData({ nodes, links });
    });
  }, []);

  const drawSankey = (width: number, data: SankeyData) => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const height = 600;

    const sankeyGen = sankey<Node, Link>()
      .nodeWidth(15)
      .nodePadding(10)
      .extent([
        [1, 1],
        [width - 1, height - 5],
      ]);

    const { nodes: layoutNodes, links: layoutLinks } = sankeyGen({
      nodes: data.nodes.map((d) => Object.assign({}, d)),
      links: data.links.map((d) => Object.assign({}, d)),
    });

    svg.attr("width", width).attr("height", height);

    svg
      .append("g")
      .selectAll("rect")
      .data(layoutNodes)
      .join("rect")
      .attr("x", (d) => d.x0 ?? 0)
      .attr("y", (d) => d.y0 ?? 0)
      .attr("height", (d) => (d.y1 ?? 0) - (d.y0 ?? 0))
      .attr("width", (d) => (d.x1 ?? 0) - (d.x0 ?? 0))
      .attr("fill", (_, i) => d3.schemeCategory10[i % 10]);

    svg
      .append("g")
      .attr("fill", "none")
      .selectAll("path")
      .data(layoutLinks)
      .join("path")
      .attr("d", sankeyLinkHorizontal())
      .attr("stroke", "#555")
      .attr("stroke-width", (d) => Math.max(1, d.width ?? 1))
      .attr("opacity", 0.6);

    svg
      .append("g")
      .selectAll("text")
      .data(layoutNodes)
      .join("text")
      .attr("x", (d) => (d.x0 ?? 0) - 6)
      .attr("y", (d) => ((d.y1 ?? 0) + (d.y0 ?? 0)) / 2)
      .attr("dy", "0.35em")
      .attr("text-anchor", "end")
      .text((d) => d.name)
      .filter((d) => (d.x0 ?? 0) < width / 2)
      .attr("x", (d) => (d.x1 ?? 0) + 6)
      .attr("text-anchor", "start");
  };

  // 当数据加载后绘图
  useEffect(() => {
    if (!data) return;
    const container = containerRef.current;
    if (!container) return;

    const draw = () => drawSankey(container.clientWidth, data);
    draw();

    const resizeObserver = new ResizeObserver(draw);
    resizeObserver.observe(container);
    return () => resizeObserver.disconnect();
  }, [data]);

  return (
    <div ref={containerRef} style={{ width: "100%", height: "600px" }}>
      <svg ref={ref}></svg>
    </div>
  );
}