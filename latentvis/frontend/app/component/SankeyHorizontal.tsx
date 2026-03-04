"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
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
        from_layer: fromAssign.layer ?? "",
        to_layer: toAssign.layer ?? "",
        from_cluster: fromAssign.cluster,
        to_cluster: toAssign.cluster,
      });
    }
  });
  return out;
}

type SankeyHorizontalProps = {
  dataPath?: string;
  secondaryDataPath?: string;
  syncTransform?: { x: number; scale: number };
  onSyncTransform?: (t: { x: number; y: number; scale: number }) => void;
  mode?: "layer" | "model";
  modelKey?: string;
  layerKey?: string;
  visibleEpochs?: string[] | null;
  reverseEpochOrder?: boolean;
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
    const fromRaw = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer;
    const toRaw = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer;
    if (!fromRaw || !toRaw) return;
    const fromKey = mode === "layer" ? canonicalLayerKey(fromRaw) : fromRaw;
    const toKey = mode === "layer" ? canonicalLayerKey(toRaw) : toRaw;
    present.add(fromKey);
    present.add(toKey);
    if (!firstSeen.has(fromKey)) firstSeen.set(fromKey, seenIdx++);
    if (!firstSeen.has(toKey)) firstSeen.set(toKey, seenIdx++);
    const ek = `${fromKey}->${toKey}`;
    edgeCounts.set(ek, (edgeCounts.get(ek) ?? 0) + 1);
  });

  const fromPreferred = preferredOrder.filter((k) => present.has(k));

  const fallback = Array.from(present).sort((a, b) => {
    const na = Number(a);
    const nb = Number(b);
    if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
    const sa = firstSeen.get(a) ?? Number.MAX_SAFE_INTEGER;
    const sb = firstSeen.get(b) ?? Number.MAX_SAFE_INTEGER;
    if (sa !== sb) return sa - sb;
    return a.localeCompare(b);
  });
  if (fromPreferred.length > 0) {
    const seen = new Set(fromPreferred);
    const remaining = fallback.filter((k) => !seen.has(k));
    return [...fromPreferred, ...remaining];
  }
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
  if (name === "fp") return "fingerprint";
  if (/^fnn_layer_/.test(name)) return name.replace(/^fnn_layer_/, "ffn_layer_");
  return name;
}

function normalizeAdjacentRows(
  rows: SampleRow[],
  mode: "layer" | "model",
  preferredOrder: string[]
): SampleRow[] {
  const rawSequence = inferSequenceFromRows(rows, mode, preferredOrder);
  const sequence =
    mode === "layer"
      ? rawSequence.filter((k, i) => rawSequence.findIndex((x) => canonicalLayerKey(x) === canonicalLayerKey(k)) === i)
      : rawSequence;
  const idxMap = new Map(
    sequence.map((k, i) => [mode === "layer" ? canonicalLayerKey(k) : k, i])
  );
  const out: SampleRow[] = [];

  rows.forEach((r) => {
    const rawFrom = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer;
    const rawTo = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer;
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
      from_layer: r.to_layer as string,
      to_layer: r.from_layer as string,
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
      : canonicalLayerKey(dir === "from" ? r.from_layer : r.to_layer);

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
        from_layer: mode === "layer" ? fromKey : (fromAssign.layer ?? rows[0]?.from_layer ?? ""),
        to_layer: mode === "layer" ? toKey : (toAssign.layer ?? rows[0]?.to_layer ?? ""),
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
// Unsynced layout tweak: negative moves left, positive moves right.
const UNSYNC_SHIFT_X = 0;
const UNSYNC_LEFT_PAD = 20;
const UNSYNC_RIGHT_PAD = 20;
const MIN_LAYER_GAP = 60;
const MIN_EPOCH_GAP = 120;

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

export default function SankeyHorizontal({
  dataPath = "/data/dimenet/sankey/sankey_samples.json",
  secondaryDataPath,
  syncTransform,
  onSyncTransform,
  mode = "layer",
  modelKey,
  layerKey,
  visibleEpochs,
  reverseEpochOrder = false,
  activeNodeKey,
  highlightId,
  highlightIds,
  onNodeSelect,
}: SankeyHorizontalProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const nodeSamplesRef = useRef<Map<string, Set<string>>>(new Map());
  const onNodeSelectRef = useRef(onNodeSelect);
  const syncTransformRef = useRef(syncTransform);
  const onSyncTransformRef = useRef(onSyncTransform);
  const activeNodeKeyRef = useRef(activeNodeKey);
  const isControlledRef = useRef(activeNodeKey !== undefined);
  const displayActiveNodeRef = useRef<string | null>(null);
  const draggingRef = useRef(false);
  const dragStartRef = useRef<{ x: number; panX: number }>({ x: 0, panX: 0 });
  const CELL_W = 120;
  const SYNC_BASE_OFFSET = 1 * CELL_W;
  const ZOOM_MIN = 0.01;
  const ZOOM_MAX = 4;
  const ZOOM_STEP = 1.15;
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [zoomX, setZoomX] = useState(1);
  const [panX, setPanX] = useState(0);
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
    let lastSecondaryRows: SampleRow[] | null = null;

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
      const visibleEpochOrder =
        mode === "model" && Array.isArray(visibleEpochs) && visibleEpochs.length > 0
          ? (reverseEpochOrder ? [...visibleEpochs].reverse() : visibleEpochs).map((x) => String(x))
          : null;
      const order =
        mode === "model" && visibleEpochOrder
          ? visibleEpochOrder
          : mode === "model"
            ? (reverseEpochOrder ? [...MODEL_ORDER].reverse() : MODEL_ORDER)
            : LAYER_ORDER;
      const filterRowsForView = (rowsInput: SampleRow[]): SampleRow[] =>
        mode === "model" && visibleEpochOrder
          ? (() => {
              const stitched = stitchRowsByVisibleEpochs(rowsInput, visibleEpochOrder);
              if (stitched.length > 0) return stitched;
              const allowed = new Set(visibleEpochOrder);
              return rowsInput.filter((r) => {
                const fromKey = String(r.from_model ?? "");
                const toKey = String(r.to_model ?? "");
                return allowed.has(fromKey) && allowed.has(toKey);
              });
            })()
          : rowsInput;
      const filteredRows = filterRowsForView(lastRows);
      const filteredSecondaryRows = lastSecondaryRows ? filterRowsForView(lastSecondaryRows) : [];
      const renderRows = normalizeAdjacentRows(filteredRows, mode, order);
      const syntheticRows = synthesizeMissingAdjacentRows(renderRows, mode, order);
      const graphRows = syntheticRows.length > 0 ? [...renderRows, ...syntheticRows] : renderRows;
      const secondaryRenderRows = normalizeAdjacentRows(filteredSecondaryRows, mode, order);
      const secondarySyntheticRows = synthesizeMissingAdjacentRows(secondaryRenderRows, mode, order);
      const secondaryGraphRows =
        secondarySyntheticRows.length > 0 ? [...secondaryRenderRows, ...secondarySyntheticRows] : secondaryRenderRows;
      const combinedGraphRows = secondaryGraphRows.length > 0 ? [...graphRows, ...secondaryGraphRows] : graphRows;
      const graph = buildGraph(combinedGraphRows);

      const NODE_W = 12;
      const CELL_W = 120;
      const CELL_MARGIN_LEFT = 10;
      const INNER_W = 100;
      const LEGEND_LEFT = 32;
      const MARGIN_LEFT = 50 + LEGEND_LEFT;
      const TOP = 10;
      const BOTTOM = 26;

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
      const axisRows = combinedGraphRows.length > 0 ? combinedGraphRows : (filteredRows.length > 0 ? filteredRows : graphRows);
      const normalizeAxisKey = (k: string) => (mode === "layer" ? canonicalLayerKey(k) : k);
      const presentLayers = new Set<string>();
      axisRows.forEach((r) => {
        const fromKey = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer;
        const toKey = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer;
        presentLayers.add(normalizeAxisKey(fromKey));
        presentLayers.add(normalizeAxisKey(toKey));
      });
      const inferredLayersRaw = inferSequenceFromRows(axisRows, mode, order).map((l) => normalizeAxisKey(l));
      const inferredLayers = inferredLayersRaw.filter((l, i) => presentLayers.has(l) && inferredLayersRaw.indexOf(l) === i);
      const layers =
        mode === "model" && visibleEpochOrder
          ? visibleEpochOrder
          : inferredLayers;
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
      const spacing = isSynced ? CELL_W : unsyncedStep;
      const totalWidth = isSynced
        ? width
        : Math.max(
            width,
            MARGIN_LEFT + centerOffset + (layers.length - 0.5) * unsyncedStep + offsetX + NODE_W / 2 + 10
          );

      // re-create svg with expanded width if needed
      svg.attr("width", totalWidth);

      // Layer/model labels for each column (bottom side).
      const labelY = height - 16;
      svg
        .append("g")
        .attr("class", "axis-labels")
        .selectAll("text")
        .data(layers)
        .enter()
        .append("text")
        .attr("x", (_d, i) => {
          const baseX = isSynced
            ? MARGIN_LEFT + centerOffset + (i + 0.5) * spacing
            : layers.length <= 1
              ? width / 2
              : UNSYNC_LEFT_PAD + i * unsyncedStep;
          return baseX + offsetX;
        })
        .attr("y", labelY)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "ideographic")
        .attr("font-size", 12)
        .attr("font-weight", 600)
        .attr("fill", "#374151")
        .attr("stroke", "#ffffff")
        .attr("stroke-width", 2)
        .attr("paint-order", "stroke")
        .style("pointer-events", "none")
        .text((d) => toShortLayerName(String(d)));

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
      const shouldRenderPaths = true;
      const focusGroup = activeLayer ?? (mode === "model" ? modelKey ?? null : layerKey ?? null);
      const clusterColor = (cluster: number) =>
        CLUSTER_COLORS[Math.max(0, cluster) % CLUSTER_COLORS.length];
      const darker = (hex: string) => {
        const c = d3.color(hex);
        return c ? c.darker(0.6).formatHex() : hex;
      };

      const secondaryNodeValue = new Map<string, number>();
      secondaryGraphRows.forEach((r) => {
        const fromKey = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer;
        const toKey = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer;
        const srcName = `${fromKey}_${r.from_cluster}`;
        const tgtName = `${toKey}_${r.to_cluster}`;
        secondaryNodeValue.set(srcName, (secondaryNodeValue.get(srcName) ?? 0) + 1);
        secondaryNodeValue.set(tgtName, (secondaryNodeValue.get(tgtName) ?? 0) + 1);
      });

      const linkGroup = svg.append("g").attr("class", "noaug-links");
      const sampleGroup = svg.append("g").attr("class", "sample-trajectories");
      const nodeGroup = svg.append("g").attr("class", "noaug-nodes");

      if (shouldRenderPaths) {
        linkGroup
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
      }

      // build per-sample trajectories with per-node y offsets (spread inside node height)
      const nodeSamples = new Map<string, Set<string>>();
      combinedGraphRows.forEach((r) => {
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
      combinedGraphRows.forEach((r) => {
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
          .attr("stroke-width", 1.2)
          .attr("opacity", 0)
          .attr("class", (d) => `sample-${d.id}`)
          .attr("data-id", (d) => d.id);
      }

      const nodeSel = nodeGroup
        .selectAll("rect.node-main")
        .data(sankeyGraph.nodes as SankeyNode<any, any>[])
        .enter()
        .append("rect")
        .attr("class", "node-main")
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

      nodeGroup
        .selectAll("rect.node-test")
        .data(sankeyGraph.nodes as SankeyNode<any, any>[])
        .enter()
        .append("rect")
        .attr("class", "node-test")
        .attr("x", (d: any) => d.x0)
        .attr("y", (d: any) => {
          const name = String(d.name || "");
          const total = Math.max(0, Number(d.value ?? 0));
          const test = Math.max(0, Number(secondaryNodeValue.get(name) ?? 0));
          const ratio = total > 0 ? Math.min(1, test / total) : 0;
          const h = Math.max(1, (d.y1 ?? 0) - (d.y0 ?? 0));
          const testH = h * ratio;
          return (d.y0 ?? 0) + Math.max(0, h - testH);
        })
        .attr("width", (d: any) => d.x1 - d.x0)
        .attr("height", (d: any) => {
          const name = String(d.name || "");
          const total = Math.max(0, Number(d.value ?? 0));
          const test = Math.max(0, Number(secondaryNodeValue.get(name) ?? 0));
          const ratio = total > 0 ? Math.min(1, test / total) : 0;
          const h = Math.max(1, (d.y1 ?? 0) - (d.y0 ?? 0));
          return Math.max(0, h * ratio);
        })
        .attr("fill", (d: any) => {
          const name = String(d.name || "");
          if (focusGroup) {
            const idx = name.lastIndexOf("_");
            const layer = idx === -1 ? name : name.slice(0, idx);
            const cluster = Number(idx === -1 ? 0 : name.slice(idx + 1));
            if (layer === focusGroup && Number.isFinite(cluster)) return darker(clusterColor(cluster));
          }
          return darker(SANKEY_NODE_COLOR);
        })
        .attr("data-name", (d: any) => d.name)
        .style("pointer-events", "none");

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
            return isHovered ? 2.2 : 1.2;
          });
      }

      // Re-apply current transform right after redraw so viewport state is preserved.
      const groups = svg.selectAll(
        "g.noaug-links, g.sample-trajectories, g.noaug-nodes, g.cluster-number-labels, g.axis-labels"
      );
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

    Promise.all([
      fetch(dataPath).then((res) => {
        if (!res.ok) throw new Error("Failed to fetch noaug_sankey_samples.json");
        return res.json();
      }),
      secondaryDataPath
        ? fetch(secondaryDataPath)
            .then((res) => (res.ok ? res.json() : null))
            .catch(() => null)
        : Promise.resolve(null),
    ])
      .then(([json, secondaryJson]: [SampleRow[], SampleRow[] | null]) => {
        lastRows = json;
        lastSecondaryRows = Array.isArray(secondaryJson) ? secondaryJson : null;
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
        const nextPanX = currSync.x - SYNC_BASE_OFFSET * currSync.scale - delta;
        onSyncTransformRef.current?.({ x: nextPanX + SYNC_BASE_OFFSET * currSync.scale, y: 0, scale: currSync.scale });
      } else {
        host.scrollLeft += delta;
      }
    }

    container.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      container.removeEventListener("wheel", onWheel as EventListener);
      ro.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataPath, secondaryDataPath, mode, modelKey, layerKey, visibleEpochs, reverseEpochOrder, !!syncTransform]);

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
      const isTestRect = rect.classList.contains("node-test");
      const baseColor =
        focusGroup && layer === focusGroup && Number.isFinite(cluster)
          ? CLUSTER_COLORS[Math.max(0, cluster) % CLUSTER_COLORS.length]
          : SANKEY_NODE_COLOR;
      const c = d3.color(baseColor);
      const testColor = c ? c.darker(0.6).formatHex() : baseColor;

      rect.setAttribute("fill", isTestRect ? testColor : baseColor);

      if (isTestRect) {
        rect.setAttribute("stroke", "none");
        rect.setAttribute("stroke-width", "0");
        return;
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
      line.setAttribute("stroke-width", isHovered ? "2.2" : "1.2");
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
  }, [displayActiveNode, highlightId, highlightIds, mode, modelKey, layerKey]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const svg = container.querySelector("svg");
    if (!svg) return;
    const groups = svg.querySelectorAll(
      "g.noaug-links, g.sample-trajectories, g.noaug-nodes, g.cluster-number-labels, g.axis-labels"
    );
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

  function applyZoom(multiplier: number) {
    if (syncTransform) {
      const prevScale = syncTransform.scale;
      const nextScale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, prevScale * multiplier));
      if (nextScale === prevScale) return;
      const currOffset = syncTransform.x - SYNC_BASE_OFFSET * prevScale;
      const nextX = currOffset + SYNC_BASE_OFFSET * nextScale;
      onSyncTransform?.({ x: nextX, y: 0, scale: nextScale });
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
        style={{ overflowX: "auto", overflowY: "hidden", cursor: syncTransform ? "default" : draggingRef.current ? "grabbing" : "grab" }}
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
