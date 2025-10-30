"use client";
import * as d3 from "d3";
import { sankey, sankeyLinkHorizontal, SankeyNodeMinimal, SankeyLinkMinimal } from "d3-sankey";
import React, { useEffect, useRef } from "react";

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

export default function SankeyDiagram({ data }: { data?: SankeyData }) {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const width = 400;
    const height = 300;

    const sankeyGen = sankey<Node, Link>()
      .nodeWidth(15)
      .nodePadding(10)
      .extent([
        [1, 1],
        [width - 1, height - 5],
      ]);

    const nodes = data?.nodes ?? [
      { name: "Default A" },
      { name: "Default B" },
      { name: "Default C" },
    ];

    const links = data?.links ?? [
      { source: 0, target: 1, value: 10 },
      { source: 1, target: 2, value: 5 },
    ];

    const { nodes: layoutNodes, links: layoutLinks } = sankeyGen({
      nodes: nodes.map((d) => Object.assign({}, d)),
      links: links.map((d) => Object.assign({}, d)),
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
      .attr("fill", () => d3.schemeCategory10[Math.floor(Math.random() * 10)]);

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
  }, [data]);

  return <svg ref={ref}></svg>;
}