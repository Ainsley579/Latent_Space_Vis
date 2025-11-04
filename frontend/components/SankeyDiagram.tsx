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

    // helper to extract layer name from node name
    const getLayerName = (fullName: string) => {
      const idx = fullName.lastIndexOf("_");
      return idx === -1 ? fullName : fullName.slice(0, idx);
    };

    // Derive layer names from node names
    const layerNameSet = new Set<string>();
    layoutNodes.forEach(n => {
      const layerName = getLayerName(n.name);
      layerNameSet.add(layerName);
    });
    const layerNames = Array.from(layerNameSet).sort();

    // Assign each layer a distinct hue around the color wheel using HSL
    const layerColorMap = new Map<string, string>();
    layerNames.forEach((layerName, i) => {
      layerColorMap.set(
        layerName,
        d3.schemePastel1[i % d3.schemePastel1.length] // ✅ 不加括号，直接取数组元素
      );
    });

    // Sort nodes within each layer by y0 (top to bottom)
    const nodesByLayer = new Map<string, Node[]>();
    layerNames.forEach(layerName => {
      const nodesInLayer = layoutNodes.filter(n => getLayerName(n.name) === layerName);
      nodesInLayer.sort((a, b) => (a.y0 ?? 0) - (b.y0 ?? 0));
      nodesByLayer.set(layerName, nodesInLayer);
    });

    // Removed gradient creation code

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

    // 2. 修改 link 绘制部分为彩色宽带流样式，使用单色，添加鼠标悬停 tooltip
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
        const sourceNode = d.source as Node;
        const sourceLayer = getLayerName(sourceNode.name);
        return layerColorMap.get(sourceLayer) ?? "#000";
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