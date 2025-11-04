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
  const tooltipRef = useRef<HTMLDivElement | null>(null);
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
      .nodeWidth(5)
      .nodePadding(5)
      .extent([
        [1, 1],
        [width - 1, height - 5],
      ]);

    const { nodes: layoutNodes, links: layoutLinks } = sankeyGen({
      nodes: data.nodes.map((d) => Object.assign({}, d)),
      links: data.links.map((d) => Object.assign({}, d)),
    });

    svg.attr("width", width).attr("height", height);

    const layersMap = new Map<number, {x0: number, x1: number}>();
    layoutNodes.forEach(d => {
      if (d.x0 == null || d.x1 == null) return;
      const layerWidth = sankeyGen.nodeWidth() + sankeyGen.nodePadding();
      const layerIndex = Math.round((d.x0) / layerWidth);
      if (!layersMap.has(layerIndex)) {
        layersMap.set(layerIndex, {x0: d.x0, x1: d.x1});
      } else {
        const ext = layersMap.get(layerIndex)!;
        ext.x0 = Math.min(ext.x0, d.x0);
        ext.x1 = Math.max(ext.x1, d.x1);
      }
    });

    const layerColors = d3.schemeCategory10;
    const layerGroups = Array.from(layersMap.entries()).sort((a,b) => a[0] - b[0]);

    const defs = svg.append("defs");
    layerGroups.forEach(([layerIndex, extent]) => {
      const colorStart = layerColors[layerIndex % layerColors.length];
      const colorEnd = layerColors[(layerIndex + 1) % layerColors.length];
      defs.append("linearGradient")
        .attr("id", `gradient-layer-${layerIndex}`)
        .attr("gradientUnits", "userSpaceOnUse")
        .attr("x1", extent.x0)
        .attr("x2", extent.x1)
        .attr("y1", 0)
        .attr("y2", 0)
        .selectAll("stop")
        .data([
          {offset: "0%", color: colorStart},
          {offset: "100%", color: colorEnd}
        ])
        .join("stop")
        .attr("offset", d => d.offset)
        .attr("stop-color", d => d.color);
    });

    // 3. 修改节点绘制：移除填充矩形，改为绘制细黑线
    svg
      .append("g")
      .selectAll("line")
      .data(layoutNodes)
      .join("line")
      .attr("x1", (d) => d.x0 + sankeyGen.nodeWidth()/2 ?? 0)
      .attr("x2", (d) => d.x0 + sankeyGen.nodeWidth()/2 ?? 0)
      .attr("y1", (d) => d.y0 ?? 0)
      .attr("y2", (d) => d.y1 ?? 0)
      .attr("stroke", "black")
      .attr("stroke-width", sankeyGen.nodeWidth());

    // 2. 修改 link 绘制部分为彩色宽带流样式，使用渐变色，添加鼠标悬停 tooltip
    const tooltip = d3.select(tooltipRef.current);
    const linkGroup = svg
      .append("g")
      .attr("fill", "none")
      .attr("class", "links");

    linkGroup
      .selectAll("path")
      .data(layoutLinks)
      .join("path")
      .attr("d", sankeyLinkHorizontal())
      .attr("stroke", (d) => {
        // 计算源节点层和目标节点层
        const layerWidth = sankeyGen.nodeWidth() + sankeyGen.nodePadding();
        const sourceLayer = Math.round((d.source.x0 ?? 0) / layerWidth);
        // 使用对应层的渐变
        return `url(#gradient-layer-${sourceLayer})`;
      })
      .attr("stroke-width", (d) => Math.max(1, d.width ?? 1))
      .attr("opacity", 0.4)
      .on("mouseover", (event, d) => {
        tooltip.style("opacity", 1)
          .html(
            `<strong>Source:</strong> ${d.source.name}<br/>
             <strong>Target:</strong> ${d.target.name}<br/>
             <strong>Value:</strong> ${d.value}`
          )
          .style("left", (event.pageX) + "px")
          .style("top", (event.pageY - 400) + "px");
      })
      .on("mousemove", (event) => {
        tooltip.style("left", (event.pageX) + "px")
          .style("top", (event.pageY- 400) + "px");
      })
      .on("mouseout", () => {
        tooltip.style("opacity", 0);
      });

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
    <div ref={containerRef} style={{ width: "100%", height: "600px", position: "relative" }}>
      <svg ref={ref}></svg>
      {/* 添加 tooltip div 结构 */}
      <div
        ref={tooltipRef}
        style={{
          position: "absolute",
          pointerEvents: "none",
          backgroundColor: "rgba(0,0,0,0.7)",
          color: "#fff",
          padding: "6px 8px",
          borderRadius: "4px",
          fontSize: "12px",
          opacity: 0,
          transition: "opacity 0.2s ease",
          zIndex: 10,
        }}
      ></div>
    </div>
  );
}