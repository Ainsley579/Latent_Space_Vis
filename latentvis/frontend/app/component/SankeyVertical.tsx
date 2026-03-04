"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
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

function stitchRowsByVisibleEpochs(rows: SampleRow[], visibleEpochs: string[]): SampleRow[] {
  if (!Array.isArray(visibleEpochs) || visibleEpochs.length < 2) return [];
  const orderedEpochs = visibleEpochs.map((x) => String(x));
  const allowed = new Set(orderedEpochs);
  const assignmentById = new Map<
    string,
    Map<string, { cluster: number; model?: string; layer?: string }>
  >();

  rows.forEach((r) => {
    const fromModel = String(r.from_model ?? "");
    const toModel = String(r.to_model ?? "");
    const id = String(r.real_id);
    if (!id) return;
    if (!assignmentById.has(id)) assignmentById.set(id, new Map());
    const amap = assignmentById.get(id)!;
    if (allowed.has(fromModel)) {
      amap.set(fromModel, { cluster: r.from_cluster, model: r.from_model, layer: r.from_layer });
    }
    if (allowed.has(toModel)) {
      amap.set(toModel, { cluster: r.to_cluster, model: r.to_model, layer: r.to_layer });
    }
  });

  const out: SampleRow[] = [];
  assignmentById.forEach((amap, rid) => {
    for (let i = 0; i < orderedEpochs.length - 1; i += 1) {
      const fromEpoch = orderedEpochs[i];
      const toEpoch = orderedEpochs[i + 1];
      const fromAssign = amap.get(fromEpoch);
      const toAssign = amap.get(toEpoch);
      if (!fromAssign || !toAssign) continue;
      out.push({
        real_id: rid,
        from_model: fromEpoch,
        to_model: toEpoch,
        from_layer: fromAssign.layer,
        to_layer: toAssign.layer,
        from_cluster: fromAssign.cluster,
        to_cluster: toAssign.cluster,
      });
    }
  });
  return out;
}

type SankeyVerticalProps = {
  dataPath?: string;
  syncTransform?: { x: number; y: number; scale: number };
  onSyncTransform?: (t: { x: number; y: number; scale: number }) => void;
  mode?: "layer" | "model";
  layerKey?: string;
  modelKey?: string;
  visibleEpochs?: string[] | null;
  activeNodeKey?: string | null;
  highlightId?: string | null;
  highlightIds?: string[] | null;
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
    const fromKey = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer ?? r.from_model;
    const toKey = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer ?? r.to_model;
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

function canonicalLayerKey(name: string): string {
  if (/^fnn_layer_/.test(name)) return name.replace(/^fnn_layer_/, "ffn_layer_");
  return name;
}

function normalizeAdjacentRows(
  rows: SampleRow[],
  mode: "layer" | "model",
  preferredOrder: string[]
): SampleRow[] {
  const sequence = inferSequenceFromRows(rows, mode, preferredOrder);
  const idxMap = new Map(
    sequence.map((k, i) => [mode === "layer" ? canonicalLayerKey(k) : k, i])
  );
  const out: SampleRow[] = [];

  rows.forEach((r) => {
    const rawFrom = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer ?? r.from_model;
    const rawTo = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer ?? r.to_model;
    if (!rawFrom || !rawTo) return;
    const fromKey = mode === "layer" ? canonicalLayerKey(rawFrom) : rawFrom;
    const toKey = mode === "layer" ? canonicalLayerKey(rawTo) : rawTo;
    const fi = idxMap.get(fromKey);
    const ti = idxMap.get(toKey);
    if (fi == null || ti == null) return;
    if (Math.abs(ti - fi) !== 1) return;

    if (fi <= ti) {
      out.push({
        ...r,
        from_layer: mode === "layer" ? fromKey : r.from_layer,
        to_layer: mode === "layer" ? toKey : r.to_layer,
      });
      return;
    }

    out.push({
      ...r,
      from_layer: r.to_layer,
      to_layer: r.from_layer,
      from_model: r.to_model,
      to_model: r.from_model,
      from_cluster: r.to_cluster,
      to_cluster: r.from_cluster,
    });
  });
  return out;
}

function synthesizeMissingAdjacentRows(
  rows: SampleRow[],
  mode: "layer" | "model",
  preferredOrder: string[]
): SampleRow[] {
  const sequence = inferSequenceFromRows(rows, mode, preferredOrder);
  if (sequence.length < 2) return [];
  const keyOf = (r: SampleRow, dir: "from" | "to") =>
    mode === "model"
      ? (dir === "from" ? r.from_model ?? r.from_layer : r.to_model ?? r.to_layer)
      : (dir === "from" ? r.from_layer ?? r.from_model : r.to_layer ?? r.to_model);

  const edgePairCount = new Map<string, number>();
  const assignmentsById = new Map<
    string,
    Map<string, { cluster: number; model?: string; layer?: string }>
  >();
  rows.forEach((r) => {
    const fk = keyOf(r, "from");
    const tk = keyOf(r, "to");
    if (!fk || !tk) return;
    const pair = `${fk}->${tk}`;
    edgePairCount.set(pair, (edgePairCount.get(pair) ?? 0) + 1);
    const rid = String(r.real_id);
    if (!assignmentsById.has(rid)) assignmentsById.set(rid, new Map());
    const amap = assignmentsById.get(rid)!;
    amap.set(fk, { cluster: r.from_cluster, model: r.from_model, layer: r.from_layer });
    amap.set(tk, { cluster: r.to_cluster, model: r.to_model, layer: r.to_layer });
  });

  const out: SampleRow[] = [];
  for (let i = 0; i < sequence.length - 1; i++) {
    const fromKey = sequence[i];
    const toKey = sequence[i + 1];
    if ((edgePairCount.get(`${fromKey}->${toKey}`) ?? 0) > 0) continue;
    assignmentsById.forEach((amap, rid) => {
      const fromAssign = amap.get(fromKey);
      const toAssign = amap.get(toKey);
      if (!fromAssign || !toAssign) return;
      out.push({
        real_id: rid,
        from_layer: mode === "layer" ? fromKey : (fromAssign.layer ?? rows[0]?.from_layer),
        to_layer: mode === "layer" ? toKey : (toAssign.layer ?? rows[0]?.to_layer),
        from_model: mode === "model" ? fromKey : (fromAssign.model ?? rows[0]?.from_model),
        to_model: mode === "model" ? toKey : (toAssign.model ?? rows[0]?.to_model),
        from_cluster: fromAssign.cluster,
        to_cluster: toAssign.cluster,
      });
    });
  }
  return out;
}

const LAYER_ORDER = [
  "fingerprint",
  "gnn_layer_1",
  "gnn_layer_2",
  "gnn_layer_3",
  "gnn_layer_4",
  "gnn_layer_5",
  "gnn_layer_6",
  "ffn_layer_1",
  "ffn_layer_2",
  "fnn_layer_1",
  "fnn_layer_2",
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
const MIN_LAYER_GAP = 30;
const MIN_EPOCH_GAP = 60;
const SYNC_VERTICAL_SHIFT_RATIO = 0.5;

function toShortLayerName(name: string): string {
  if (name === "fingerprint") return "fp";
  const gnn = /^gnn_layer_(\d+)$/.exec(name);
  if (gnn) return `gnn${gnn[1]}`;
  const ffn = /^ffn_layer_(\d+)$/.exec(name);
  if (ffn) return `ffn${ffn[1]}`;
  const fnn = /^fnn_layer_(\d+)$/.exec(name);
  if (fnn) return `fnn${fnn[1]}`;
  return name;
}

export default function SankeyVertical({
  dataPath = "/data/dimenet/sankey/sankey_samples.json",
  syncTransform,
  onSyncTransform,
  mode = "model",
  layerKey,
  modelKey,
  visibleEpochs,
  activeNodeKey,
  highlightId,
  highlightIds,
  onNodeSelect,
}: SankeyVerticalProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const nodeSamplesRef = useRef<Map<string, Set<string>>>(new Map());
  const onNodeSelectRef = useRef(onNodeSelect);
  const syncTransformRef = useRef(syncTransform);
  const onSyncTransformRef = useRef(onSyncTransform);
  const activeNodeKeyRef = useRef(activeNodeKey);
  const isControlledRef = useRef(activeNodeKey !== undefined);
  const displayActiveNodeRef = useRef<string | null>(null);
  const draggingRef = useRef(false);
  const dragStartRef = useRef<{ y: number; panY: number }>({ y: 0, panY: 0 });
  const CELL_H = 150;
  const SYNC_BASE_OFFSET = 1.6 * CELL_H;
  const ZOOM_MIN = 0.01;
  const ZOOM_MAX = 4;
  const ZOOM_STEP = 1.15;
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [zoomX, setZoomX] = useState(1);
  const [panY, setPanY] = useState(0);
  const [showZoomControls, setShowZoomControls] = useState(false);
  const isControlled = activeNodeKey !== undefined;
  const displayActiveNode = isControlled ? (activeNodeKey ?? null) : activeNode;

  useEffect(() => {
    onNodeSelectRef.current = onNodeSelect;
  }, [onNodeSelect]);
  useEffect(() => {
    syncTransformRef.current = syncTransform;
  }, [syncTransform]);
  useEffect(() => {
    onSyncTransformRef.current = onSyncTransform;
  }, [onSyncTransform]);

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
    const host = container;
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
      const order =
        mode === "model" && Array.isArray(visibleEpochs) && visibleEpochs.length > 0
          ? visibleEpochs.map((x) => String(x))
          : mode === "model"
            ? MODEL_ORDER
            : LAYER_ORDER;
      const filteredRows =
        mode === "model" && Array.isArray(visibleEpochs)
          ? (() => {
              const stitched = stitchRowsByVisibleEpochs(lastRows, visibleEpochs.map((x) => String(x)));
              if (stitched.length > 0) return stitched;
              const allowed = new Set(visibleEpochs.map((x) => String(x)));
              return lastRows.filter((r) => {
                const fromKey = String(r.from_model ?? "");
                const toKey = String(r.to_model ?? "");
                return allowed.has(fromKey) && allowed.has(toKey);
              });
            })()
          : lastRows;

      const rowsReal = normalizeAdjacentRows(filteredRows, mode, order) as any[];
      const rowsSynthetic = synthesizeMissingAdjacentRows(rowsReal, mode, order) as any[];
      const rows = rowsSynthetic.length > 0 ? [...rowsReal, ...rowsSynthetic] : rowsReal;
      const NODE_W = 12;
      const NODE_PAD = 4;
      const BOTTOM = 28;
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
      const inferredOrder = inferSequenceFromRows(rows as SampleRow[], mode, order);
      const renderOrder =
        mode === "model" && Array.isArray(visibleEpochs) && visibleEpochs.length > 0
          ? visibleEpochs.map((x) => String(x))
          : inferredOrder.filter((m) => (nodesByModel.get(m)?.length ?? 0) > 0);

      // In rotated rendering, x-axis maps to screen Y (vertical direction).
      const unsyncedLeftPad = UNSYNC_TOP_PAD;
      const unsyncedRightPad = UNSYNC_BOTTOM_PAD;
      const unsyncedGap =
        renderOrder.length <= 1
          ? 0
          : Math.max(
              mode === "model" ? MIN_EPOCH_GAP : MIN_LAYER_GAP,
              (drawW - unsyncedLeftPad - unsyncedRightPad - NODE_W) / (renderOrder.length - 1)
            );

      // Synced mode keeps alignment with 2D; unsynced mode fills available width.
      const MODEL_GAP = isSynced ? CELL_H : unsyncedGap;
      const baseLeft = isSynced ? MARGIN_LEFT : unsyncedLeftPad;
      const scale = 1;
      const offsetY = isSynced ? CELL_H * SYNC_VERTICAL_SHIFT_RATIO : UNSYNC_SHIFT_Y;
      const totalHeight = Math.max(
        height,
        baseLeft + (Math.max(1, renderOrder.length) - 0.5) * MODEL_GAP + offsetY + NODE_W / 2 + 10
      );
      svg.attr("height", totalHeight);

      // Layer/model labels for each row (right side; screen-space, not rotated with data group).
      const labelYOffset = -8;
      svg
        .append("g")
        .attr("class", "axis-labels")
        .selectAll("text")
        .data(renderOrder)
        .enter()
        .append("text")
        .attr("x", width - 6)
        .attr("y", (_d, i) => baseLeft + i * MODEL_GAP + offsetY + labelYOffset)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .attr("font-size", 12)
        .attr("font-weight", 600)
        .attr("fill", "#374151")
        .attr("stroke", "#ffffff")
        .attr("stroke-width", 2)
        .attr("paint-order", "stroke")
        .style("pointer-events", "none")
        .text((d) => toShortLayerName(String(d)));

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
      const shouldRenderPaths = true;
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

      if (shouldRenderPaths) {
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
          .attr("vector-effect", "non-scaling-stroke")
          .attr("stroke-width", (d) => {
            const s = nodeByKey.get(d.src);
            if (!s) return 1;
            const sTot = outTotal.get(d.src) || 1;
            return Math.max(0.5, (d.value / sTot) * s.h);
          });
      }

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

      let sampleSel: d3.Selection<SVGLineElement, { id: string; x1: number; y1: number; x2: number; y2: number }, SVGGElement, unknown> | null = null;
      if (shouldRenderPaths) {
        sampleSel = sampleGroup
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
          .attr("stroke-width", 1.1)
          .attr("opacity", 0)
          .attr("class", (d) => `sample-${d.id}`)
          .attr("data-id", (d) => d.id);
      }

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
        const highlightSet = new Set((highlightIds ?? []).map((x) => String(x)));
        if (highlightSet.size > 0) {
          return highlightSet.has(d.id) ? 1 : 0;
        }
        if (displayActiveNode) {
          const samples = nodeSamples.get(displayActiveNode) || new Set();
          return samples.has(d.id) ? 0.8 : 0;
        }
        return 0;
      };

      if (sampleSel) {
        sampleSel
          .attr("opacity", (d) => {
            const highlightSet = new Set((highlightIds ?? []).map((x) => String(x)));
            const isHovered = !!highlightId && d.id === highlightId;
            const isSearchHit = highlightSet.has(d.id);
            return (isHovered || isSearchHit) ? 1 : baseOpacity(d);
          })
          .attr("stroke", (d) => {
            const isHovered = !!highlightId && d.id === highlightId;
            return isHovered ? HIGHLIGHT_RED : SANKEY_LINK_COLOR;
          })
          .attr("stroke-width", (d) => {
            const isHovered = !!highlightId && d.id === highlightId;
            return isHovered ? 2 : 1.1;
          });
      }

      // Keep labels above links/nodes.
      svg.selectAll("g.axis-labels").raise();

      // Re-apply current transform right after redraw so viewport state is preserved.
      const contentGroups = svg.selectAll("g.links, g.sample-trajectories, g.nodes");
      const labelGroups = svg.selectAll("g.axis-labels");
      if (syncTransform) {
        const scale = syncTransform.scale;
        const offsetX = syncTransform.y - SYNC_BASE_OFFSET * scale;
        contentGroups.attr("transform", `translate(${offsetX},0) scale(${scale},1)`);
        // Keep labels on screen-space vertical transform (not horizontal drift).
        labelGroups.attr("transform", `translate(0,${offsetX}) scale(1,${scale})`);
      } else if (panY !== 0 || zoomX !== 1) {
        contentGroups.attr("transform", `translate(${panY},0) scale(${zoomX},1)`);
        labelGroups.attr("transform", `translate(0,${panY}) scale(1,${zoomX})`);
      } else {
        contentGroups.attr("transform", null);
        labelGroups.attr("transform", null);
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
      const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
      if (!Number.isFinite(delta) || delta === 0) return;
      e.preventDefault();
      const currSync = syncTransformRef.current;
      if (currSync) {
        const nextPanY = currSync.y - SYNC_BASE_OFFSET * currSync.scale - delta;
        onSyncTransformRef.current?.({ x: 0, y: nextPanY + SYNC_BASE_OFFSET * currSync.scale, scale: currSync.scale });
      } else {
        const canNativeScroll = host.scrollHeight - host.clientHeight > 1;
        if (canNativeScroll) {
          host.scrollTop += delta;
        } else {
          // Fallback for fixed-size SVG: still allow wheel to move vertically.
          setPanY((prev) => prev - delta);
        }
      }
    }

    container.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      container.removeEventListener("wheel", onWheel as EventListener);
      ro.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataPath, mode, modelKey, layerKey, visibleEpochs, !!syncTransform]);

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
    const highlightSet = new Set((highlightIds ?? []).map((x) => String(x)));
    const lines = Array.from(svg.querySelectorAll("g.sample-trajectories line[data-id]")) as SVGLineElement[];
    lines.forEach((line) => {
      const id = line.getAttribute("data-id") || "";
      const isHovered = !!highlightId && id === highlightId;
      const isSearchHit = highlightSet.has(id);
      const opacity = highlightSet.size > 0
        ? (isSearchHit ? 1 : 0)
        : isHovered
          ? 1
          : activeName
            ? (samplesForActive.has(id) ? 0.8 : 0)
            : 0;
      line.setAttribute("opacity", String(opacity));
      line.setAttribute("stroke", isHovered ? HIGHLIGHT_RED : SANKEY_LINK_COLOR);
      line.setAttribute("stroke-width", isHovered ? "2" : "1.1");
    });
  }, [displayActiveNode, highlightId, highlightIds, mode, modelKey, layerKey]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const svg = container.querySelector("svg");
    if (!svg) return;
    const contentGroups = svg.querySelectorAll("g.links, g.sample-trajectories, g.nodes");
    const labelGroups = svg.querySelectorAll("g.axis-labels");
    let contentTransformAttr = "";
    let labelTransformAttr = "";
    if (syncTransform) {
      const scale = syncTransform.scale;
      const offsetX = syncTransform.y - SYNC_BASE_OFFSET * scale;
      contentTransformAttr = `translate(${offsetX},0) scale(${scale},1)`;
      labelTransformAttr = `translate(0,${offsetX}) scale(1,${scale})`;
    } else if (panY !== 0 || zoomX !== 1) {
      // In rotated space, x-translate maps to on-screen vertical movement.
      contentTransformAttr = `translate(${panY},0) scale(${zoomX},1)`;
      labelTransformAttr = `translate(0,${panY}) scale(1,${zoomX})`;
    }

    contentGroups.forEach((el) => {
      if (contentTransformAttr) (el as SVGGElement).setAttribute("transform", contentTransformAttr);
      else (el as SVGGElement).removeAttribute("transform");
    });
    labelGroups.forEach((el) => {
      if (labelTransformAttr) (el as SVGGElement).setAttribute("transform", labelTransformAttr);
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

  function applyZoom(multiplier: number) {
    if (syncTransform) {
      const prevScale = syncTransform.scale;
      const nextScale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, prevScale * multiplier));
      if (nextScale === prevScale) return;
      const currOffset = syncTransform.y - SYNC_BASE_OFFSET * prevScale;
      const nextY = currOffset + SYNC_BASE_OFFSET * nextScale;
      onSyncTransform?.({ x: 0, y: nextY, scale: nextScale });
      return;
    }
    setZoomX((prev) => {
      const next = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, prev * multiplier));
      return next === prev ? prev : next;
    });
  }

  const zoomBtnBaseStyle: CSSProperties = {
    width: 34,
    height: 34,
    border: "none",
    background: "rgba(255,255,255,0.96)",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#111827",
    cursor: "pointer",
    transition: "background 0.12s ease, transform 0.12s ease",
  };

  return (
    <div
      className="w-full h-full relative"
      onMouseEnter={() => setShowZoomControls(true)}
      onMouseLeave={() => setShowZoomControls(false)}
    >
      <div
        ref={containerRef}
        className="w-full h-full"
        style={{ overflowX: "auto", overflowY: "scroll", cursor: syncTransform ? "default" : draggingRef.current ? "grabbing" : "grab" }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      />
      <div
        style={{
          position: "absolute",
          right: 8,
          bottom: 8,
          display: "flex",
          gap: 0,
          zIndex: 30,
          pointerEvents: showZoomControls ? "auto" : "none",
          opacity: showZoomControls ? 1 : 0,
          transition: "opacity 0.15s ease",
          border: "1px solid #d4d4d8",
          borderRadius: 8,
          overflow: "hidden",
          boxShadow: "0 2px 8px rgba(0,0,0,0.18)",
          background: "rgba(255,255,255,0.9)",
        }}
      >
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            applyZoom(ZOOM_STEP);
          }}
          title="Zoom in"
          aria-label="Zoom in"
          style={{ ...zoomBtnBaseStyle, borderRight: "1px solid #e4e4e7" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "#f3f4f6";
            e.currentTarget.style.transform = "translateY(-1px)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "rgba(255,255,255,0.96)";
            e.currentTarget.style.transform = "translateY(0)";
          }}
        >
          <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
            <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </button>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            applyZoom(1 / ZOOM_STEP);
          }}
          title="Zoom out"
          aria-label="Zoom out"
          style={zoomBtnBaseStyle}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "#f3f4f6";
            e.currentTarget.style.transform = "translateY(-1px)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "rgba(255,255,255,0.96)";
            e.currentTarget.style.transform = "translateY(0)";
          }}
        >
          <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
            <path d="M3 8h10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </button>
      </div>
    </div>
  );
}
