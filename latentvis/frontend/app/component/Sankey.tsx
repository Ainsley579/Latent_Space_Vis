"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { sankey, sankeyLinkHorizontal, SankeyLink, SankeyNode } from "d3-sankey";

type SampleRow = {
  real_id: string | number;
  from_layer: string;
  to_layer: string;
  from_cluster: number;
  to_cluster: number;
};

type NoaugSankeyProps = {
  dataPath?: string;
};

export default function NoaugSankey({ dataPath = "/data/sankey/noaug_sankey_samples.json" }: NoaugSankeyProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [activeNode, setActiveNode] = useState<string | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    container.innerHTML = "";

    const ro = new ResizeObserver(() => draw());
    ro.observe(container);

    let lastRows: SampleRow[] | null = null;

    function buildGraph(rows: SampleRow[]) {
      const nodesMap = new Map<string, number>();
      const nodes: { name: string }[] = [];
      const linksMap = new Map<string, number>();

      rows.forEach((r) => {
        const src = `${r.from_layer}_${r.from_cluster}`;
        const tgt = `${r.to_layer}_${r.to_cluster}`;
        if (!nodesMap.has(src)) {
          nodesMap.set(src, nodes.length);
          nodes.push({ name: src });
        }
        if (!nodesMap.has(tgt)) {
          nodesMap.set(tgt, nodes.length);
          nodes.push({ name: tgt });
        }
        const key = `${src}->${tgt}`;
        linksMap.set(key, (linksMap.get(key) ?? 0) + 1);
      });

      const links: Array<{ source: number; target: number; value: number; key: string }> = Array.from(
        linksMap.entries()
      ).map(([key, value]) => {
        const [src, tgt] = key.split("->");
        return {
          source: nodesMap.get(src)!,
          target: nodesMap.get(tgt)!,
          value,
          key,
        };
      });

      return { nodes, links };
    }

    function draw() {
      if (!container) return;
      if (!lastRows) return;

      container.innerHTML = "";
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);

      const graph = buildGraph(lastRows);

      const svg = d3.select(container).append("svg").attr("width", width).attr("height", height);

      const sankeyGen = sankey<{ name: string }, { source: number; target: number; value: number; key: string }>()
        .nodeWidth(12)
        .nodePadding(10)
        .extent([
          [10, 10],
          [width - 10, height - 10],
        ]);

      const sankeyGraph = sankeyGen({
        nodes: graph.nodes.map((d) => ({ ...d })),
        links: graph.links.map((d) => ({ ...d })),
      });

      const nodeByName = new Map<string, any>();
      (sankeyGraph.nodes as SankeyNode<any, any>[]).forEach((n: any) => {
        nodeByName.set(String(n.name), n);
      });

      const linkGroup = svg.append("g").attr("class", "noaug-links");
      const nodeGroup = svg.append("g").attr("class", "noaug-nodes");
      const sampleGroup = svg.append("g").attr("class", "sample-trajectories");

      const linkSel = linkGroup
        .selectAll("path")
        .data(sankeyGraph.links as SankeyLink<any, any>[])
        .enter()
        .append("path")
        .attr("d", sankeyLinkHorizontal())
        .attr("fill", "none")
        .attr("stroke", "#8aa1b3")
        .attr("stroke-opacity", 0.4)
        .attr("stroke-width", (d: any) => Math.max(1, d.width || 1))
        .attr("data-key", (d: any) => d.key);

      // build per-sample trajectories with per-node y offsets (spread inside node height)
      const nodeSamples = new Map<string, Set<string>>();
      lastRows.forEach((r) => {
        const srcName = `${r.from_layer}_${r.from_cluster}`;
        const tgtName = `${r.to_layer}_${r.to_cluster}`;
        const id = String(r.real_id);
        if (!nodeSamples.has(srcName)) nodeSamples.set(srcName, new Set());
        if (!nodeSamples.has(tgtName)) nodeSamples.set(tgtName, new Set());
        nodeSamples.get(srcName)!.add(id);
        nodeSamples.get(tgtName)!.add(id);
      });

      const sampleNodeYMap = new Map<string, Map<string, number>>();
      (sankeyGraph.nodes as SankeyNode<any, any>[]).forEach((node: any) => {
        const name = String(node.name || "");
        const samples = Array.from(nodeSamples.get(name) || []);
        if (samples.length === 0) return;
        samples.sort((a, b) => {
          const na = Number(a), nb = Number(b);
          if (!isNaN(na) && !isNaN(nb)) return na - nb;
          return a < b ? -1 : a > b ? 1 : 0;
        });
        const y0 = node.y0 ?? 0;
        const y1 = node.y1 ?? 0;
        const h = Math.max(1, y1 - y0);
        samples.forEach((id, i) => {
          const y = samples.length === 1 ? y0 + h / 2 : y0 + (i + 0.5) * (h / samples.length);
          if (!sampleNodeYMap.has(id)) sampleNodeYMap.set(id, new Map());
          sampleNodeYMap.get(id)!.set(name, y);
        });
      });

      const sampleLines: Array<{ id: string; x1: number; y1: number; x2: number; y2: number }> = [];
      lastRows.forEach((r) => {
        const srcName = `${r.from_layer}_${r.from_cluster}`;
        const tgtName = `${r.to_layer}_${r.to_cluster}`;
        const src = nodeByName.get(srcName);
        const tgt = nodeByName.get(tgtName);
        if (!src || !tgt) return;
        const id = String(r.real_id);
        const y1 = sampleNodeYMap.get(id)?.get(srcName);
        const y2 = sampleNodeYMap.get(id)?.get(tgtName);
        if (y1 == null || y2 == null) return;
        const x1 = src.x1;
        const x2 = tgt.x0;
        sampleLines.push({ id, x1, y1, x2, y2 });
      });

      const sampleSel = sampleGroup
        .selectAll("line")
        .data(sampleLines)
        .enter()
        .append("line")
        .attr("x1", (d) => d.x1)
        .attr("y1", (d) => d.y1)
        .attr("x2", (d) => d.x2)
        .attr("y2", (d) => d.y2)
        .attr("stroke", "#d97706")
        .attr("stroke-width", 1.2)
        .attr("opacity", 0)
        .attr("class", (d) => `sample-${d.id}`);

      const nodeSel = nodeGroup
        .selectAll("rect")
        .data(sankeyGraph.nodes as SankeyNode<any, any>[])
        .enter()
        .append("rect")
        .attr("x", (d: any) => d.x0)
        .attr("y", (d: any) => d.y0)
        .attr("width", (d: any) => d.x1 - d.x0)
        .attr("height", (d: any) => Math.max(1, d.y1 - d.y0))
        .attr("fill", "#c9d6df")
        .attr("stroke", "#6b7c8a")
        .attr("data-name", (d: any) => d.name)
        .style("cursor", "pointer")
        .on("click", (_event: any, d: any) => {
          const name = String(d.name || "");
          setActiveNode((prev) => (prev === name ? null : name));
        });

      nodeSel.append("title").text((d: any) => String(d.name || ""));

      if (activeNode) {
        const samples = nodeSamples.get(activeNode) || new Set();
        sampleSel.attr("opacity", (d) => (samples.has(d.id) ? 0.8 : 0));
      } else {
        sampleSel.attr("opacity", 0);
      }
    }

    fetch(dataPath)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch noaug_sankey_samples.json");
        return res.json();
      })
      .then((json: SampleRow[]) => {
        lastRows = json;
        draw();
      })
      .catch((err) => {
        console.error(err);
        if (container) container.innerHTML = "<div style='color:red'>Failed to load sankey data.</div>";
      });

    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataPath, activeNode]);

  return <div ref={containerRef} className="w-full h-full" />;
}
