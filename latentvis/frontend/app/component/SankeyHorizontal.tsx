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

type SankeyHorizontalProps = {
  dataPath?: string;
  syncTransform?: { x: number; scale: number };
  mode?: "layer" | "model";
  modelKey?: string;
  layerKey?: string;
  activeNodeKey?: string | null;
  onNodeSelect?: (sel: { model?: string; layer?: string; cluster: number; ids?: string[] }) => void;
};

const LAYER_ORDER = [
  "emb",
  "output_block_0",
  "output_block_1",
  "output_block_2",
  "output_block_3",
  "output_block_4",
  "u_mlp_input",
  "u_mlp_l1",
  "u_mlp_l2",
];

const MODEL_ORDER = ["dimenet10conf","dimenet_noaug"];

export default function SankeyHorizontal({
  dataPath = "/data/sankey/noaug_sankey_samples.json",
  syncTransform,
  mode = "layer",
  modelKey,
  layerKey,
  activeNodeKey,
  onNodeSelect,
}: SankeyHorizontalProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [zoomX, setZoomX] = useState(1);
  const displayActiveNode = activeNodeKey ?? activeNode;

  useEffect(() => {
    if (activeNodeKey == null) {
      setActiveNode(null);
    }
  }, [activeNodeKey]);

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

      const svg = d3.select(container).append("svg").attr("width", width).attr("height", height);
      const graph = buildGraph(lastRows);

      const NODE_W = 12;
      const CELL_W = 120;
      const CELL_MARGIN_LEFT = 10;
      const INNER_W = 100;
      const LEGEND_LEFT = 32;
      const MARGIN_LEFT = 50 + LEGEND_LEFT;
      const TOP = 10;
      const BOTTOM = 10;

      const sankeyGen = sankey<{ name: string }, { source: number; target: number; value: number; key: string }>()
        .nodeWidth(NODE_W)
        .nodePadding(5)
        .extent([
          [10, TOP],
          [width - 10, height - BOTTOM],
        ]);

      const sankeyGraph = sankeyGen({
        nodes: graph.nodes.map((d) => ({ ...d })),
        links: graph.links.map((d) => ({ ...d })),
      });

      // enforce x positions by layer order to align with SankeyView columns
      const presentLayers = new Set<string>();
      lastRows.forEach((r) => {
        presentLayers.add(r.from_layer);
        presentLayers.add(r.to_layer);
      });
      const layers = LAYER_ORDER.filter((l) => presentLayers.has(l));
      const layerIndex = new Map(layers.map((l, i) => [l, i]));
      const centerOffset = CELL_MARGIN_LEFT + INNER_W / 2;
      const scaleX = syncTransform?.scale ?? zoomX;
      const offsetX = (syncTransform?.x ?? 0) - 1 * CELL_W * scaleX;
      const spacing = CELL_W * scaleX;
      const totalWidth = Math.max(
        width,
        scaleX * (MARGIN_LEFT + centerOffset + (layers.length - 0.5) * CELL_W) + offsetX + NODE_W / 2 + 10
      );

      // re-create svg with expanded width if needed
      svg.attr("width", totalWidth);

      (sankeyGraph.nodes as SankeyNode<any, any>[]).forEach((node: any) => {
        const name = String(node.name || "");
        const idxC = name.lastIndexOf("_");
        const layer = idxC === -1 ? name : name.slice(0, idxC);
        const idx = layerIndex.get(layer);
        if (idx == null) return;
        const baseX = MARGIN_LEFT + centerOffset + (idx + 0.5) * CELL_W;
        const centerX = scaleX * baseX + offsetX;
        node.x0 = centerX - NODE_W / 2;
        node.x1 = node.x0 + NODE_W;
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
        .attr("stroke", "#4a7bc4ff")
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
        .attr("fill", (d: any) => (displayActiveNode && String(d.name) === displayActiveNode ? "#f4b266" : "#c9d6df"))
        .attr("stroke", (d: any) => (displayActiveNode && String(d.name) === displayActiveNode ? "#c27d2b" : "#6b7c8a"))
        .attr("data-name", (d: any) => d.name)
        .style("cursor", "pointer")
        .on("click", (event: any, d: any) => {
          event.stopPropagation();
          const name = String(d.name || "");
          const next = displayActiveNode === name ? null : name;
          if (activeNodeKey == null) {
            setActiveNode(next);
          }
          const idx = name.lastIndexOf("_");
          const layer = idx === -1 ? name : name.slice(0, idx);
          const cluster = Number(idx === -1 ? 0 : name.slice(idx + 1));
          const ids = Array.from(nodeSamples.get(name) || []);
          onNodeSelect?.({ model: modelKey, layer, cluster, ids });
        });

      nodeSel.append("title").text((d: any) => String(d.name || ""));

      if (displayActiveNode) {
        const samples = nodeSamples.get(displayActiveNode) || new Set();
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

    function onWheel(e: WheelEvent) {
      if (syncTransform) return;
      e.preventDefault();
      const next = Math.max(0.5, Math.min(4, zoomX * (1 + -e.deltaY * 0.002)));
      if (next !== zoomX) setZoomX(next);
    }

    container.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      container.removeEventListener("wheel", onWheel as EventListener);
      ro.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataPath, activeNode, activeNodeKey, zoomX, syncTransform]);

  return <div ref={containerRef} className="w-full h-full" style={{ overflowX: "auto", overflowY: "hidden" }} />;
}
