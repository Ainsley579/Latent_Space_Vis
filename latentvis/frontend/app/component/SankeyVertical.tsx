"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { sankey, sankeyLinkHorizontal, SankeyLink, SankeyNode } from "d3-sankey";
import { CLUSTER_COLORS, SANKEY_NODE_COLOR, SANKEY_LINK_COLOR, HIGHLIGHT_RED } from "../theme/colors";

type SampleRow = {
  real_id: string | number;
  from_layer?: string;
  to_layer?: string;
  from_model?: string;
  to_model?: string;
  from_cluster: number;
  to_cluster: number;
};

type SankeyVerticalProps = {
  dataPath?: string;
  syncTransform?: { x: number; y: number; scale: number };
  onSyncTransform?: (t: { x: number; y: number; scale: number }) => void;
  mode?: "layer" | "model";
  layerKey?: string;
  modelKey?: string;
  activeNodeKey?: string | null;
  highlightId?: string | null;
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
// Unsynced layout tweak: negative moves up, positive moves down (after rotated render).
const UNSYNC_SHIFT_Y = 0;
const UNSYNC_TOP_PAD = 14;
const UNSYNC_BOTTOM_PAD = 14;

export default function SankeyVertical({
  dataPath = "/data/dimenet/sankey/sankey_samples.json",
  syncTransform,
  onSyncTransform,
  mode = "model",
  layerKey,
  modelKey,
  activeNodeKey,
  highlightId,
  onNodeSelect,
}: SankeyVerticalProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const nodeSamplesRef = useRef<Map<string, Set<string>>>(new Map());
  const onNodeSelectRef = useRef(onNodeSelect);
  const activeNodeKeyRef = useRef(activeNodeKey);
  const isControlledRef = useRef(activeNodeKey !== undefined);
  const displayActiveNodeRef = useRef<string | null>(null);
  const draggingRef = useRef(false);
  const dragStartRef = useRef<{ y: number; panY: number }>({ y: 0, panY: 0 });
  const CELL_H = 150;
  const SYNC_BASE_OFFSET = 1.6 * CELL_H;
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [zoomX, setZoomX] = useState(1);
  const [panY, setPanY] = useState(0);
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
      const TOP = 10;
      const MARGIN_LEFT = CELL_H * 2 + 10;
      const bandH = Math.max(1, drawH - TOP - BOTTOM);

      const nodeValue = new Map<string, number>();
      rows.forEach((r) => {
        const fromKey = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer ?? r.from_model;
        const toKey = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer ?? r.to_model;
        const src = `${fromKey}_${r.from_cluster}`;
        const tgt = `${toKey}_${r.to_cluster}`;
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

      const isSynced = !!syncTransform;
      const order = mode === "model" ? MODEL_ORDER : LAYER_ORDER;
      const presentOrder = order.filter((m) => (nodesByModel.get(m)?.length ?? 0) > 0);
      const fallbackOrder = Array.from(nodesByModel.keys()).sort((a, b) => {
        const na = Number(a);
        const nb = Number(b);
        if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
        return a.localeCompare(b);
      });
      const renderOrder = presentOrder.length > 0 ? presentOrder : fallbackOrder;

      // In rotated rendering, x-axis maps to screen Y (vertical direction).
      const unsyncedLeftPad = UNSYNC_TOP_PAD;
      const unsyncedRightPad = UNSYNC_BOTTOM_PAD;
      const unsyncedGap =
        renderOrder.length <= 1
          ? 0
          : Math.max(40, (drawW - unsyncedLeftPad - unsyncedRightPad - NODE_W) / (renderOrder.length - 1));

      // Synced mode keeps alignment with 2D; unsynced mode fills available width.
      const MODEL_GAP = isSynced ? CELL_H : unsyncedGap;
      const baseLeft = isSynced ? MARGIN_LEFT : unsyncedLeftPad;
      const scale = 1;
      const offsetY = isSynced ? 0 : UNSYNC_SHIFT_Y;

      const nodes: Array<{ key: string; model: string; value: number; x: number; y: number; h: number }> = [];
      renderOrder.forEach((model, i) => {
        const list = nodesByModel.get(model) || [];
        list.sort((a, b) => a.cluster - b.cluster);
        const total = list.reduce((s, n) => s + n.value, 0) || 1;
        const totalPad = Math.min(bandH * 0.5, Math.max(0, list.length - 1) * NODE_PAD);
        const availH = Math.max(1, bandH - totalPad);
        let cur = TOP;
        list.forEach((n) => {
          const h = (n.value / total) * availH;
          const baseX = baseLeft + i * MODEL_GAP;
          const x = baseX + offsetY;
          nodes.push({ key: n.key, model, value: n.value, x, y: cur, h });
          cur += h + NODE_PAD;
        });
      });

      let activeModel: string | null = null;
      if (displayActiveNode) {
        const idx = displayActiveNode.lastIndexOf("_");
        activeModel = idx === -1 ? displayActiveNode : displayActiveNode.slice(0, idx);
      }
      const focusGroup = activeModel ?? (mode === "model" ? modelKey ?? null : layerKey ?? null);

      const clusterValues: number[] = [];
      nodes.forEach((n) => {
        if (!activeModel || n.model !== activeModel) return;
        const idx = n.key.lastIndexOf("_");
        const cluster = Number(idx === -1 ? 0 : n.key.slice(idx + 1));
        if (Number.isFinite(cluster)) clusterValues.push(cluster);
      });
      const maxCluster = clusterValues.length ? Math.max(...clusterValues) : 0;
      // const color = d3.scaleOrdinal<number, string>(d3.schemeTableau10).domain(d3.range(maxCluster + 1));

      const color = d3
        .scaleOrdinal<number, string>()
        .domain(d3.range(maxCluster + 1))
        .range(CLUSTER_COLORS);

      const nodeByKey = new Map(nodes.map((n) => [n.key, n]));
      const linkMap = new Map<string, number>();
      rows.forEach((r) => {
        const fromKey = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer ?? r.from_model;
        const toKey = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer ?? r.to_model;
        const src = `${fromKey}_${r.from_cluster}`;
        const tgt = `${toKey}_${r.to_cluster}`;
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
      const sampleGroup = g.append("g").attr("class", "sample-trajectories");
      const nodeGroup = g.append("g").attr("class", "nodes");

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
        .attr("stroke", SANKEY_LINK_COLOR)
        .attr("stroke-opacity", 0.35)
        .attr("stroke-width", (d) => {
          const s = nodeByKey.get(d.src);
          if (!s) return 1;
          const sTot = outTotal.get(d.src) || 1;
          return Math.max(0.5, (d.value / sTot) * s.h);
        });

      const nodeSamples = new Map<string, Set<string>>();
      rows.forEach((r) => {
        const fromKey = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer ?? r.from_model;
        const toKey = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer ?? r.to_model;
        const src = `${fromKey}_${r.from_cluster}`;
        const tgt = `${toKey}_${r.to_cluster}`;
        const id = String(r.real_id);
        if (!nodeSamples.has(src)) nodeSamples.set(src, new Set());
        if (!nodeSamples.has(tgt)) nodeSamples.set(tgt, new Set());
        nodeSamples.get(src)!.add(id);
        nodeSamples.get(tgt)!.add(id);
      });
      nodeSamplesRef.current = nodeSamples;

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
        const fromKey = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer ?? r.from_model;
        const toKey = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer ?? r.to_model;
        const src = `${fromKey}_${r.from_cluster}`;
        const tgt = `${toKey}_${r.to_cluster}`;
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
        .attr("stroke", SANKEY_LINK_COLOR)
        .attr("stroke-width", 1.1)
        .attr("opacity", 0)
        .attr("class", (d) => `sample-${d.id}`)
        .attr("data-id", (d) => d.id);

      const nodeSel = nodeGroup
        .selectAll("rect")
        .data(nodes)
        .enter()
        .append("rect")
        .attr("x", (d) => d.x)
        .attr("y", (d) => d.y)
        .attr("width", NODE_W)
        .attr("height", (d) => Math.max(1, d.h))
        .attr("fill", (d) => {
          if (focusGroup && d.model === focusGroup) {
            const idx = d.key.lastIndexOf("_");
            const cluster = Number(idx === -1 ? 0 : d.key.slice(idx + 1));
            if (Number.isFinite(cluster)) return color(Math.max(0, cluster));
          }
          return SANKEY_NODE_COLOR;
        })
        .attr("stroke", (d) => (displayActiveNode && d.key === displayActiveNode ? HIGHLIGHT_RED : "none"))
        .attr("stroke-width", (d) => (displayActiveNode && d.key === displayActiveNode ? 2 : 0))
        .attr("data-key", (d) => d.key)
        .style("cursor", "pointer")
        .on("click", (event, d) => {
          event.stopPropagation();
          const currentDisplay = displayActiveNodeRef.current;
          const next = currentDisplay === d.key ? null : d.key;
          if (!isControlledRef.current) {
            setActiveNode(next);
          }
          const idx = d.key.lastIndexOf("_");
          const keyName = idx === -1 ? d.key : d.key.slice(0, idx);
          const cluster = Number(idx === -1 ? 0 : d.key.slice(idx + 1));
          const ids = Array.from(nodeSamples.get(d.key) || []);
          if (mode === "model") {
            onNodeSelectRef.current?.({ model: keyName, layer: layerKey, cluster, ids });
          } else {
            onNodeSelectRef.current?.({ model: modelKey, layer: keyName, cluster, ids });
          }
        });

      nodeSel.append("title").text((d) => d.key);
      nodeGroup.raise();

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
        .attr("stroke-width", (d) => (highlightId && d.id === highlightId ? 2 : 1.1));

      // Re-apply current transform right after redraw so viewport state is preserved.
      const groups = svg.selectAll("g.links, g.sample-trajectories, g.nodes");
      if (syncTransform) {
        const scale = syncTransform.scale;
        const offsetX = syncTransform.y - SYNC_BASE_OFFSET * scale;
        groups.attr("transform", `translate(${offsetX},0) scale(${scale},1)`);
      } else if (panY !== 0 || zoomX !== 1) {
        groups.attr("transform", `translate(${panY},0) scale(${zoomX},1)`);
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
        const currOffset = syncTransform.y - SYNC_BASE_OFFSET * prevScale;
        const nextY = currOffset + SYNC_BASE_OFFSET * nextScale;
        onSyncTransform?.({ x: 0, y: nextY, scale: nextScale });
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
    const activeModel = activeName ? activeName.slice(0, Math.max(0, activeName.lastIndexOf("_"))) : null;
    const focusGroup = activeModel ?? (mode === "model" ? modelKey ?? null : layerKey ?? null);

    const rects = Array.from(svg.querySelectorAll("g.nodes rect[data-key]")) as SVGRectElement[];
    rects.forEach((rect) => {
      const key = rect.getAttribute("data-key") || "";
      const idx = key.lastIndexOf("_");
      const model = idx === -1 ? key : key.slice(0, idx);
      const cluster = Number(idx === -1 ? 0 : key.slice(idx + 1));

      if (focusGroup && model === focusGroup && Number.isFinite(cluster)) {
        rect.setAttribute("fill", CLUSTER_COLORS[Math.max(0, cluster) % CLUSTER_COLORS.length]);
      } else {
        rect.setAttribute("fill", SANKEY_NODE_COLOR);
      }

      if (activeName && key === activeName) {
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
      line.setAttribute("stroke-width", isHighlight ? "2" : "1.1");
    });
  }, [displayActiveNode, highlightId, mode, modelKey, layerKey]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const svg = container.querySelector("svg");
    if (!svg) return;
    const groups = svg.querySelectorAll("g.links, g.sample-trajectories, g.nodes");
    let transformAttr = "";
    if (syncTransform) {
      const scale = syncTransform.scale;
      const offsetX = syncTransform.y - SYNC_BASE_OFFSET * scale;
      transformAttr = `translate(${offsetX},0) scale(${scale},1)`;
    } else if (panY !== 0 || zoomX !== 1) {
      // In rotated space, x-translate maps to on-screen vertical movement.
      transformAttr = `translate(${panY},0) scale(${zoomX},1)`;
    }

    groups.forEach((el) => {
      if (transformAttr) (el as SVGGElement).setAttribute("transform", transformAttr);
      else (el as SVGGElement).removeAttribute("transform");
    });
  }, [syncTransform, panY, zoomX]);

  function onMouseDown(e: React.MouseEvent<HTMLDivElement>) {
    if (e.button !== 0) return;
    draggingRef.current = true;
    const startPanY = syncTransform ? syncTransform.y - SYNC_BASE_OFFSET * syncTransform.scale : panY;
    dragStartRef.current = { y: e.clientY, panY: startPanY };
  }

  function onMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    if (!draggingRef.current) return;
    const dy = e.clientY - dragStartRef.current.y;
    const nextPanY = dragStartRef.current.panY + dy;
    if (syncTransform) {
      onSyncTransform?.({ x: 0, y: nextPanY + SYNC_BASE_OFFSET * syncTransform.scale, scale: syncTransform.scale });
    } else {
      setPanY(nextPanY);
    }
  }

  function onMouseUp() {
    draggingRef.current = false;
  }

  return (
    <div
      ref={containerRef}
      className="w-full h-full"
      style={{ overflowX: "auto", overflowY: "hidden", cursor: syncTransform ? "default" : draggingRef.current ? "grabbing" : "grab" }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
    />
  );
}
