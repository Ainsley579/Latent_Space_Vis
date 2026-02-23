"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { sankey, sankeyLinkHorizontal, SankeyLink, SankeyNode } from "d3-sankey";
import { CLUSTER_COLORS, SANKEY_NODE_COLOR, SANKEY_LINK_COLOR, HIGHLIGHT_RED } from "../theme/colors";

type SampleRow = {
  real_id: string | number;
  from_layer: string;
  to_layer: string;
  from_model?: string;
  to_model?: string;
  from_cluster: number;
  to_cluster: number;
};

type SankeyHorizontalProps = {
  dataPath?: string;
  syncTransform?: { x: number; scale: number };
  onSyncTransform?: (t: { x: number; y: number; scale: number }) => void;
  mode?: "layer" | "model";
  modelKey?: string;
  layerKey?: string;
  activeNodeKey?: string | null;
  highlightId?: string | null;
  onNodeSelect?: (sel: { model?: string; layer?: string; cluster: number; ids?: string[] }) => void;
};

function inferSequenceFromRows(
  rows: SampleRow[],
  mode: "layer" | "model",
  preferredOrder: string[]
): string[] {
  const firstSeen = new Map<string, number>();
  let seenIdx = 0;
  const present = new Set<string>();
  const edgeCounts = new Map<string, number>();

  rows.forEach((r) => {
    const fromKey = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer;
    const toKey = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer;
    if (!fromKey || !toKey) return;
    present.add(fromKey);
    present.add(toKey);
    if (!firstSeen.has(fromKey)) firstSeen.set(fromKey, seenIdx++);
    if (!firstSeen.has(toKey)) firstSeen.set(toKey, seenIdx++);
    const ek = `${fromKey}->${toKey}`;
    edgeCounts.set(ek, (edgeCounts.get(ek) ?? 0) + 1);
  });

  const fromPreferred = preferredOrder.filter((k) => present.has(k));
  if (fromPreferred.length > 0) return fromPreferred;

  const fallback = Array.from(present).sort((a, b) => {
    const na = Number(a);
    const nb = Number(b);
    if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
    const sa = firstSeen.get(a) ?? Number.MAX_SAFE_INTEGER;
    const sb = firstSeen.get(b) ?? Number.MAX_SAFE_INTEGER;
    if (sa !== sb) return sa - sb;
    return a.localeCompare(b);
  });
  if (mode !== "layer") return fallback;

  const bestNext = new Map<string, string>();
  const bestVal = new Map<string, number>();
  const incoming = new Set<string>();
  edgeCounts.forEach((cnt, key) => {
    const [src, tgt] = key.split("->");
    if (!src || !tgt || src === tgt) return;
    const prev = bestVal.get(src) ?? -1;
    if (cnt > prev) {
      bestVal.set(src, cnt);
      bestNext.set(src, tgt);
    }
  });
  bestNext.forEach((tgt) => incoming.add(tgt));

  const starts = fallback.filter((k) => !incoming.has(k));
  const order: string[] = [];
  const visited = new Set<string>();
  const walk = (start: string) => {
    let cur: string | undefined = start;
    while (cur && !visited.has(cur)) {
      visited.add(cur);
      order.push(cur);
      cur = bestNext.get(cur);
    }
  };
  starts.forEach(walk);
  fallback.forEach(walk);
  return order;
}

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
// Unsynced layout tweak: negative moves left, positive moves right.
const UNSYNC_SHIFT_X = 0;
const UNSYNC_LEFT_PAD = 20;
const UNSYNC_RIGHT_PAD = 20;
const MIN_LAYER_GAP = 80;
const MIN_EPOCH_GAP = 150;
const SYNC_EPOCH_GAP = 170;

export default function SankeyHorizontal({
  dataPath = "/data/dimenet/sankey/sankey_samples.json",
  syncTransform,
  onSyncTransform,
  mode = "layer",
  modelKey,
  layerKey,
  activeNodeKey,
  highlightId,
  onNodeSelect,
}: SankeyHorizontalProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const nodeSamplesRef = useRef<Map<string, Set<string>>>(new Map());
  const onNodeSelectRef = useRef(onNodeSelect);
  const activeNodeKeyRef = useRef(activeNodeKey);
  const isControlledRef = useRef(activeNodeKey !== undefined);
  const displayActiveNodeRef = useRef<string | null>(null);
  const draggingRef = useRef(false);
  const dragStartRef = useRef<{ x: number; panX: number }>({ x: 0, panX: 0 });
  const CELL_W = 120;
  const SYNC_BASE_OFFSET = 1 * CELL_W;
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [zoomX, setZoomX] = useState(1);
  const [panX, setPanX] = useState(0);
  const isControlled = activeNodeKey !== undefined;
  const displayActiveNode = isControlled ? (activeNodeKey ?? null) : activeNode;

  useEffect(() => {
    onNodeSelectRef.current = onNodeSelect;
  }, [onNodeSelect]);

  useEffect(() => {
    activeNodeKeyRef.current = activeNodeKey;
  }, [activeNodeKey]);

  useEffect(() => {
    isControlledRef.current = activeNodeKey !== undefined;
  }, [activeNodeKey]);

  useEffect(() => {
    displayActiveNodeRef.current = displayActiveNode;
  }, [displayActiveNode]);

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
        const fromKey = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer;
        const toKey = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer;
        const src = `${fromKey}_${r.from_cluster}`;
        const tgt = `${toKey}_${r.to_cluster}`;
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
        const fromKey = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer;
        const toKey = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer;
        presentLayers.add(fromKey);
        presentLayers.add(toKey);
      });
      const order = mode === "model" ? MODEL_ORDER : LAYER_ORDER;
      const layers = inferSequenceFromRows(lastRows, mode, order).filter((l) => presentLayers.has(l));
      const layerIndex = new Map(layers.map((l, i) => [l, i]));
      const centerOffset = CELL_MARGIN_LEFT + INNER_W / 2;
      const isSynced = !!syncTransform;
      const scaleX = 1;
      const unsyncedStep = (() => {
        if (isSynced) return CELL_W;
        if (layers.length <= 1) return 0;
        const minGap = mode === "model" ? MIN_EPOCH_GAP : MIN_LAYER_GAP;
        return Math.max(minGap, (width - UNSYNC_LEFT_PAD - UNSYNC_RIGHT_PAD) / (layers.length - 1));
      })();
      const offsetX = isSynced ? 0 : UNSYNC_SHIFT_X;
      const spacing = isSynced ? (mode === "model" ? SYNC_EPOCH_GAP : CELL_W) : unsyncedStep;
      const totalWidth = isSynced
        ? width
        : Math.max(
            width,
            MARGIN_LEFT + centerOffset + (layers.length - 0.5) * unsyncedStep + offsetX + NODE_W / 2 + 10
          );

      // re-create svg with expanded width if needed
      svg.attr("width", totalWidth);

      (sankeyGraph.nodes as SankeyNode<any, any>[]).forEach((node: any) => {
        const name = String(node.name || "");
        const idxC = name.lastIndexOf("_");
        const layer = idxC === -1 ? name : name.slice(0, idxC);
        const idx = layerIndex.get(layer);
        if (idx == null) return;
        const baseX = isSynced
          ? MARGIN_LEFT + centerOffset + (idx + 0.5) * spacing
          : (layers.length <= 1 ? width / 2 : UNSYNC_LEFT_PAD + idx * unsyncedStep);
        const centerX = baseX + offsetX;
        node.x0 = centerX - NODE_W / 2;
        node.x1 = node.x0 + NODE_W;
      });

      const nodeByName = new Map<string, any>();
      (sankeyGraph.nodes as SankeyNode<any, any>[]).forEach((n: any) => {
        nodeByName.set(String(n.name), n);
      });

      let activeLayer: string | null = null;
      if (displayActiveNode) {
        const idx = displayActiveNode.lastIndexOf("_");
        activeLayer = idx === -1 ? displayActiveNode : displayActiveNode.slice(0, idx);
      }
      const focusGroup = activeLayer ?? (mode === "model" ? modelKey ?? null : layerKey ?? null);
      const clusterColor = (cluster: number) =>
        CLUSTER_COLORS[Math.max(0, cluster) % CLUSTER_COLORS.length];

      const linkGroup = svg.append("g").attr("class", "noaug-links");
      const sampleGroup = svg.append("g").attr("class", "sample-trajectories");
      const nodeGroup = svg.append("g").attr("class", "noaug-nodes");

      const linkSel = linkGroup
        .selectAll("path")
        .data(sankeyGraph.links as SankeyLink<any, any>[])
        .enter()
        .append("path")
        .attr("d", sankeyLinkHorizontal())
        .attr("fill", "none")
        .attr("stroke", SANKEY_LINK_COLOR)
        .attr("stroke-opacity", 0.4)
        .attr("vector-effect", "non-scaling-stroke")
        .attr("stroke-width", (d: any) => Math.max(1, d.width || 1))
        .attr("data-key", (d: any) => d.key);

      // build per-sample trajectories with per-node y offsets (spread inside node height)
      const nodeSamples = new Map<string, Set<string>>();
      lastRows.forEach((r) => {
        const fromKey = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer;
        const toKey = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer;
        const srcName = `${fromKey}_${r.from_cluster}`;
        const tgtName = `${toKey}_${r.to_cluster}`;
        const id = String(r.real_id);
        if (!nodeSamples.has(srcName)) nodeSamples.set(srcName, new Set());
        if (!nodeSamples.has(tgtName)) nodeSamples.set(tgtName, new Set());
        nodeSamples.get(srcName)!.add(id);
        nodeSamples.get(tgtName)!.add(id);
      });
      nodeSamplesRef.current = nodeSamples;

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
        const fromKey = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer;
        const toKey = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer;
        const srcName = `${fromKey}_${r.from_cluster}`;
        const tgtName = `${toKey}_${r.to_cluster}`;
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
        .attr("stroke", SANKEY_LINK_COLOR)
        .attr("vector-effect", "non-scaling-stroke")
        .attr("stroke-width", 1.2)
        .attr("opacity", 0)
        .attr("class", (d) => `sample-${d.id}`)
        .attr("data-id", (d) => d.id);

      const nodeSel = nodeGroup
        .selectAll("rect")
        .data(sankeyGraph.nodes as SankeyNode<any, any>[])
        .enter()
        .append("rect")
        .attr("x", (d: any) => d.x0)
        .attr("y", (d: any) => d.y0)
        .attr("width", (d: any) => d.x1 - d.x0)
        .attr("height", (d: any) => Math.max(1, d.y1 - d.y0))
        .attr("fill", (d: any) => {
          if (focusGroup) {
            const name = String(d.name || "");
            const idx = name.lastIndexOf("_");
            const layer = idx === -1 ? name : name.slice(0, idx);
            const cluster = Number(idx === -1 ? 0 : name.slice(idx + 1));
            if (layer === focusGroup && Number.isFinite(cluster)) return clusterColor(cluster);
          }
          return SANKEY_NODE_COLOR;
        })
        .attr("stroke", (d: any) => (displayActiveNode && String(d.name) === displayActiveNode ? HIGHLIGHT_RED : "none"))
        .attr("stroke-width", (d: any) => (displayActiveNode && String(d.name) === displayActiveNode ? 2 : 0))
        .attr("data-name", (d: any) => d.name)
        .style("cursor", "pointer")
        .on("click", (event: any, d: any) => {
          event.stopPropagation();
          const name = String(d.name || "");
          const currentDisplay = displayActiveNodeRef.current;
          const next = currentDisplay === name ? null : name;
          if (!isControlledRef.current) {
            setActiveNode(next);
          }
          const idx = name.lastIndexOf("_");
          const keyName = idx === -1 ? name : name.slice(0, idx);
          const cluster = Number(idx === -1 ? 0 : name.slice(idx + 1));
          const ids = Array.from(nodeSamples.get(name) || []);
          if (mode === "model") {
            onNodeSelectRef.current?.({ model: keyName, layer: layerKey, cluster, ids });
          } else {
            onNodeSelectRef.current?.({ model: modelKey, layer: keyName, cluster, ids });
          }
        });

      nodeSel.append("title").text((d: any) => String(d.name || ""));
      nodeGroup.raise();

      // When a cluster is activated, annotate cluster numbers on the same layer/model.
      if (displayActiveNode && activeLayer) {
        const labelData = (sankeyGraph.nodes as SankeyNode<any, any>[])
          .map((n: any) => {
            const name = String(n.name || "");
            const idx = name.lastIndexOf("_");
            const layer = idx === -1 ? name : name.slice(0, idx);
            const cluster = Number(idx === -1 ? NaN : name.slice(idx + 1));
            if (layer !== activeLayer || !Number.isFinite(cluster)) return null;
            return { node: n, cluster, isActive: name === displayActiveNode };
          })
          .filter((d): d is { node: any; cluster: number; isActive: boolean } => !!d);

        const labelGroup = svg.append("g").attr("class", "cluster-number-labels").style("pointer-events", "none");
        labelGroup
          .selectAll("text")
          .data(labelData)
          .enter()
          .append("text")
          .attr("x", (d) => (d.node.x1 ?? 0) + 6)
          .attr("y", (d) => ((d.node.y0 ?? 0) + (d.node.y1 ?? 0)) / 2 + 4)
          .attr("font-size", 11)
          .attr("font-weight", (d) => (d.isActive ? 800 : 700))
          .attr("fill", (d) => (d.isActive ? HIGHLIGHT_RED : "#1f2937"))
          .attr("stroke", "#ffffff")
          .attr("stroke-width", 2.4)
          .attr("paint-order", "stroke")
          .text((d) => String(d.cluster));
      }

      const baseOpacity = (d: { id: string }) => {
        if (displayActiveNode) {
          const samples = nodeSamples.get(displayActiveNode) || new Set();
          return samples.has(d.id) ? 0.8 : 0;
        }
        return 0;
      };

      sampleSel
        .attr("opacity", (d) => (highlightId && d.id === highlightId ? 1 : baseOpacity(d)))
        .attr("stroke", (d) => (highlightId && d.id === highlightId ? HIGHLIGHT_RED : SANKEY_LINK_COLOR))
        .attr("stroke-width", (d) => (highlightId && d.id === highlightId ? 2.2 : 1.2));

      // Re-apply current transform right after redraw so viewport state is preserved.
      const groups = svg.selectAll("g.noaug-links, g.sample-trajectories, g.noaug-nodes, g.cluster-number-labels");
      if (syncTransform) {
        const scale = syncTransform.scale;
        const offsetX = syncTransform.x - SYNC_BASE_OFFSET * scale;
        groups.attr("transform", `translate(${offsetX},0) scale(${scale},1)`);
      } else if (panX !== 0 || zoomX !== 1) {
        groups.attr("transform", `translate(${panX},0) scale(${zoomX},1)`);
      } else {
        groups.attr("transform", null);
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
      e.preventDefault();
      if (syncTransform) {
        const prevScale = syncTransform.scale;
        const nextScale = Math.max(0.01, Math.min(4, prevScale * (1 + -e.deltaY * 0.002)));
        if (nextScale === prevScale) return;
        const currOffset = syncTransform.x - SYNC_BASE_OFFSET * prevScale;
        const nextX = currOffset + SYNC_BASE_OFFSET * nextScale;
        onSyncTransform?.({ x: nextX, y: 0, scale: nextScale });
      } else {
        setZoomX((prev) => {
          const next = Math.max(0.01, Math.min(4, prev * (1 + -e.deltaY * 0.002)));
          return next === prev ? prev : next;
        });
      }
    }

    container.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      container.removeEventListener("wheel", onWheel as EventListener);
      ro.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataPath, mode, modelKey, layerKey, !!syncTransform]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const svg = container.querySelector("svg");
    if (!svg) return;

    const activeName = displayActiveNode;
    const activeLayer = activeName ? activeName.slice(0, Math.max(0, activeName.lastIndexOf("_"))) : null;
    const focusGroup = activeLayer ?? (mode === "model" ? modelKey ?? null : layerKey ?? null);

    const rects = Array.from(svg.querySelectorAll("g.noaug-nodes rect[data-name]")) as SVGRectElement[];
    rects.forEach((rect) => {
      const name = rect.getAttribute("data-name") || "";
      const idx = name.lastIndexOf("_");
      const layer = idx === -1 ? name : name.slice(0, idx);
      const cluster = Number(idx === -1 ? 0 : name.slice(idx + 1));

      if (focusGroup && layer === focusGroup && Number.isFinite(cluster)) {
        rect.setAttribute("fill", CLUSTER_COLORS[Math.max(0, cluster) % CLUSTER_COLORS.length]);
      } else {
        rect.setAttribute("fill", SANKEY_NODE_COLOR);
      }

      if (activeName && name === activeName) {
        rect.setAttribute("stroke", HIGHLIGHT_RED);
        rect.setAttribute("stroke-width", "2");
      } else {
        rect.setAttribute("stroke", "none");
        rect.setAttribute("stroke-width", "0");
      }
    });

    const samplesForActive = activeName ? nodeSamplesRef.current.get(activeName) || new Set<string>() : new Set<string>();
    const lines = Array.from(svg.querySelectorAll("g.sample-trajectories line[data-id]")) as SVGLineElement[];
    lines.forEach((line) => {
      const id = line.getAttribute("data-id") || "";
      const isHighlight = !!highlightId && id === highlightId;
      const opacity = isHighlight ? 1 : activeName ? (samplesForActive.has(id) ? 0.8 : 0) : 0;
      line.setAttribute("opacity", String(opacity));
      line.setAttribute("stroke", isHighlight ? HIGHLIGHT_RED : SANKEY_LINK_COLOR);
      line.setAttribute("stroke-width", isHighlight ? "2.2" : "1.2");
    });

    const svgSel = d3.select(svg);
    svgSel.selectAll("g.cluster-number-labels").remove();
    if (activeLayer) {
      const labels = rects
        .map((rect) => {
          const name = rect.getAttribute("data-name") || "";
          const idx = name.lastIndexOf("_");
          const layer = idx === -1 ? name : name.slice(0, idx);
          const cluster = Number(idx === -1 ? NaN : name.slice(idx + 1));
          if (layer !== activeLayer || !Number.isFinite(cluster)) return null;
          const x = Number(rect.getAttribute("x") || 0);
          const y = Number(rect.getAttribute("y") || 0);
          const w = Number(rect.getAttribute("width") || 0);
          const h = Number(rect.getAttribute("height") || 0);
          return { cluster, isActive: name === activeName, x, y, w, h };
        })
        .filter((d): d is { cluster: number; isActive: boolean; x: number; y: number; w: number; h: number } => !!d);

      const labelGroup = svgSel.append("g").attr("class", "cluster-number-labels").style("pointer-events", "none");
      labelGroup
        .selectAll("text")
        .data(labels)
        .enter()
        .append("text")
        .attr("x", (d) => d.x + d.w + 6)
        .attr("y", (d) => d.y + d.h / 2 + 4)
        .attr("font-size", 11)
        .attr("font-weight", (d) => (d.isActive ? 800 : 700))
        .attr("fill", (d) => (d.isActive ? HIGHLIGHT_RED : "#1f2937"))
        .attr("stroke", "#ffffff")
        .attr("stroke-width", 2.4)
        .attr("paint-order", "stroke")
        .text((d) => String(d.cluster));
    }
  }, [displayActiveNode, highlightId, mode, modelKey, layerKey]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const svg = container.querySelector("svg");
    if (!svg) return;
    const groups = svg.querySelectorAll("g.noaug-links, g.sample-trajectories, g.noaug-nodes, g.cluster-number-labels");
    let transformAttr = "";
    if (syncTransform) {
      const scale = syncTransform.scale;
      const offsetX = syncTransform.x - SYNC_BASE_OFFSET * scale;
      transformAttr = `translate(${offsetX},0) scale(${scale},1)`;
    } else if (panX !== 0 || zoomX !== 1) {
      transformAttr = `translate(${panX},0) scale(${zoomX},1)`;
    }

    groups.forEach((el) => {
      if (transformAttr) (el as SVGGElement).setAttribute("transform", transformAttr);
      else (el as SVGGElement).removeAttribute("transform");
    });
  }, [syncTransform, panX, zoomX]);

  function onMouseDown(e: React.MouseEvent<HTMLDivElement>) {
    if (e.button !== 0) return;
    draggingRef.current = true;
    const startPanX = syncTransform ? syncTransform.x - SYNC_BASE_OFFSET * syncTransform.scale : panX;
    dragStartRef.current = { x: e.clientX, panX: startPanX };
  }

  function onMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    if (!draggingRef.current) return;
    const dx = e.clientX - dragStartRef.current.x;
    const nextPanX = dragStartRef.current.panX + dx;
    if (syncTransform) {
      onSyncTransform?.({ x: nextPanX + SYNC_BASE_OFFSET * syncTransform.scale, y: 0, scale: syncTransform.scale });
    } else {
      setPanX(nextPanX);
    }
  }

  function onMouseUp() {
    draggingRef.current = false;
  }

  return (
    <div
      ref={containerRef}
      className="w-full h-full"
      style={{ overflowX: "hidden", overflowY: "hidden", cursor: syncTransform ? "default" : draggingRef.current ? "grabbing" : "grab" }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
    />
  );
}
