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

type SankeyVerticalProps = {
  dataPath?: string;
  syncTransform?: { x: number; y: number; scale: number };
  mode?: "layer" | "model";
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

const MODEL_ORDER = ["dimenet_noaug", "dimenet10conf"];

export default function SankeyVertical({
  dataPath = "/data/sankey/noaug_sankey_samples.json",
  syncTransform,
  mode = "layer",
  layerKey,
  activeNodeKey,
  onNodeSelect,
}: SankeyVerticalProps) {
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
      // draw in swapped space, then rotate +90deg to fit container
      const drawW = height;
      const drawH = width;

      const svg = d3.select(container).append("svg").attr("width", width).attr("height", height);
      const g = svg.append("g").attr("transform", `translate(${width},0) rotate(90)`);

      // model transition within a single layer
      const rows = lastRows as any[];
      const NODE_W = 12;
      const NODE_PAD = 4;
      const BOTTOM = 10;
      const CELL_H = 150;
      const TOP = 10;
      const MARGIN_LEFT = CELL_H * 2 + 10;
      const bandH = Math.max(1, drawH - TOP - BOTTOM);
      // model spacing should match cell height
      const MODEL_GAP = CELL_H;
      const scale = syncTransform?.scale ?? 1;
      const offsetY = (syncTransform?.y ?? 0) - 1 * CELL_H * scale;

      const nodeValue = new Map<string, number>();
      rows.forEach((r) => {
        const src = `${r.from_model}_${r.from_cluster}`;
        const tgt = `${r.to_model}_${r.to_cluster}`;
        nodeValue.set(src, (nodeValue.get(src) ?? 0) + 1);
        nodeValue.set(tgt, (nodeValue.get(tgt) ?? 0) + 1);
      });

      const nodesByModel = new Map<string, Array<{ key: string; cluster: number; value: number }>>();
      nodeValue.forEach((value, key) => {
        const idx = key.lastIndexOf("_");
        const model = idx === -1 ? key : key.slice(0, idx);
        const cluster = Number(idx === -1 ? 0 : key.slice(idx + 1));
        if (!nodesByModel.has(model)) nodesByModel.set(model, []);
        nodesByModel.get(model)!.push({ key, cluster, value });
      });

      const nodes: Array<{ key: string; model: string; value: number; x: number; y: number; h: number }> = [];
      MODEL_ORDER.forEach((model, i) => {
        const list = nodesByModel.get(model) || [];
        list.sort((a, b) => a.cluster - b.cluster);
        const total = list.reduce((s, n) => s + n.value, 0) || 1;
        const totalPad = Math.min(bandH * 0.5, Math.max(0, list.length - 1) * NODE_PAD);
        const availH = Math.max(1, bandH - totalPad);
        let cur = TOP;
        list.forEach((n) => {
          const h = (n.value / total) * availH;
          const baseX = i === 0 ? MARGIN_LEFT : MARGIN_LEFT + MODEL_GAP;
          const x = baseX * scale + offsetY;
          nodes.push({ key: n.key, model, value: n.value, x, y: cur, h });
          cur += h + NODE_PAD;
        });
      });

      const nodeByKey = new Map(nodes.map((n) => [n.key, n]));
      const linkMap = new Map<string, number>();
      rows.forEach((r) => {
        const src = `${r.from_model}_${r.from_cluster}`;
        const tgt = `${r.to_model}_${r.to_cluster}`;
        const k = `${src}->${tgt}`;
        linkMap.set(k, (linkMap.get(k) ?? 0) + 1);
      });

      const links = Array.from(linkMap.entries()).map(([k, value]) => {
        const [src, tgt] = k.split("->");
        return { src, tgt, value };
      });

      const outOffset = new Map<string, number>();
      const inOffset = new Map<string, number>();
      const outTotal = new Map<string, number>();
      const inTotal = new Map<string, number>();
      links.forEach((d) => {
        outTotal.set(d.src, (outTotal.get(d.src) ?? 0) + d.value);
        inTotal.set(d.tgt, (inTotal.get(d.tgt) ?? 0) + d.value);
      });

      const linkGroup = g.append("g").attr("class", "links");
      const nodeGroup = g.append("g").attr("class", "nodes");
      const sampleGroup = g.append("g").attr("class", "sample-trajectories");

      linkGroup
        .selectAll("path")
        .data(links)
        .enter()
        .append("path")
        .attr("d", (d) => {
          const s = nodeByKey.get(d.src);
          const t = nodeByKey.get(d.tgt);
          if (!s || !t) return "";
          const sTot = outTotal.get(d.src) || 1;
          const tTot = inTotal.get(d.tgt) || 1;
          const sOff = outOffset.get(d.src) || 0;
          const tOff = inOffset.get(d.tgt) || 0;
          const y1 = s.y + ((sOff + d.value / 2) / sTot) * s.h;
          const y2 = t.y + ((tOff + d.value / 2) / tTot) * t.h;
          outOffset.set(d.src, sOff + d.value);
          inOffset.set(d.tgt, tOff + d.value);
          const x1 = s.x <= t.x ? s.x + NODE_W : t.x + NODE_W;
          const x2 = s.x <= t.x ? t.x : s.x;
          const dx = Math.abs(x2 - x1) * 0.5;
          return `M ${x1},${y1} C ${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
        })
        .attr("fill", "none")
        .attr("stroke", "#8aa1b3")
        .attr("stroke-opacity", 0.35)
        .attr("stroke-width", (d) => {
          const s = nodeByKey.get(d.src);
          if (!s) return 1;
          const sTot = outTotal.get(d.src) || 1;
          return Math.max(0.5, (d.value / sTot) * s.h);
        });

      const nodeSamples = new Map<string, Set<string>>();
      rows.forEach((r) => {
        const src = `${r.from_model}_${r.from_cluster}`;
        const tgt = `${r.to_model}_${r.to_cluster}`;
        const id = String(r.real_id);
        if (!nodeSamples.has(src)) nodeSamples.set(src, new Set());
        if (!nodeSamples.has(tgt)) nodeSamples.set(tgt, new Set());
        nodeSamples.get(src)!.add(id);
        nodeSamples.get(tgt)!.add(id);
      });

      const sampleNodeY = new Map<string, Map<string, number>>();
      nodes.forEach((node) => {
        const samples = Array.from(nodeSamples.get(node.key) || []);
        if (samples.length === 0) return;
        samples.sort((a, b) => Number(a) - Number(b));
        samples.forEach((id, i) => {
          const y = node.h <= 1 ? node.y : node.y + (i + 0.5) * (node.h / samples.length);
          if (!sampleNodeY.has(id)) sampleNodeY.set(id, new Map());
          sampleNodeY.get(id)!.set(node.key, y);
        });
      });

      const sampleLines = rows.flatMap((r) => {
        const src = `${r.from_model}_${r.from_cluster}`;
        const tgt = `${r.to_model}_${r.to_cluster}`;
        const s = nodeByKey.get(src);
        const t = nodeByKey.get(tgt);
        if (!s || !t) return [];
        const id = String(r.real_id);
        const y1 = sampleNodeY.get(id)?.get(src);
        const y2 = sampleNodeY.get(id)?.get(tgt);
        if (y1 == null || y2 == null) return [];
        return [{ id, x1: s.x + NODE_W, y1, x2: t.x, y2 }];
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
        .attr("stroke", "#4a7bc4")
        .attr("stroke-width", 1.1)
        .attr("opacity", 0)
        .attr("class", (d) => `sample-${d.id}`);

      const nodeSel = nodeGroup
        .selectAll("rect")
        .data(nodes)
        .enter()
        .append("rect")
        .attr("x", (d) => d.x)
        .attr("y", (d) => d.y)
        .attr("width", NODE_W)
        .attr("height", (d) => Math.max(1, d.h))
        .attr("fill", (d) => (displayActiveNode && d.key === displayActiveNode ? "#f4b266" : "#c9d6df"))
        .attr("stroke", (d) => (displayActiveNode && d.key === displayActiveNode ? "#c27d2b" : "#6b7c8a"))
        .style("cursor", "pointer")
        .on("click", (event, d) => {
          event.stopPropagation();
          const next = displayActiveNode === d.key ? null : d.key;
          if (activeNodeKey == null) {
            setActiveNode(next);
          }
          const idx = d.key.lastIndexOf("_");
          const model = idx === -1 ? d.key : d.key.slice(0, idx);
          const cluster = Number(idx === -1 ? 0 : d.key.slice(idx + 1));
          const ids = Array.from(nodeSamples.get(d.key) || []);
          onNodeSelect?.({ model, layer: layerKey, cluster, ids });
        });

      nodeSel.append("title").text((d) => d.key);

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
