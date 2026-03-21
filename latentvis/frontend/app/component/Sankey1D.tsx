"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import * as d3 from "d3";
import { sankey, sankeyLinkHorizontal, SankeyLink, SankeyNode } from "d3-sankey";
import { CLUSTER_COLORS, SANKEY_NODE_COLOR, SANKEY_LINK_COLOR, HIGHLIGHT_RED } from "../theme/colors";
import { fetchJsonCached } from "../utils/fetchCache";

type SampleRow = {
    real_id: string | number;
    from_layer?: string;
    to_layer?: string;
    from_model?: string;
    to_model?: string;
    from_cluster: number;
    to_cluster: number;
    split?: SplitTag;
};

function rowsFromCompactOverview(
    overview: any,
    showFfn2: boolean,
    selectedSplits?: string[],
    mode?: "layer" | "model",
    modelKey?: string,
    layerKey?: string
): SampleRow[] {
    const epochs: string[] = Array.isArray(overview?.epochs) ? overview.epochs.map((x: any) => String(x)) : [];
    const layersRaw: string[] = Array.isArray(overview?.layers) ? overview.layers.map((x: any) => String(x)) : [];
    const layers = showFfn2 ? layersRaw : layersRaw.filter((layer) => normalizeLayerForVisibility(layer) !== HIDDEN_LAYER);
    const cells: any[] = Array.isArray(overview?.cells) ? overview.cells : [];
    if (epochs.length === 0 || layers.length === 0 || cells.length === 0) return [];

    const layerIndexByName = new Map(layersRaw.map((name, idx) => [name, idx]));
    const labelByMoleculeId: Record<string, string> = {};
    if (Array.isArray(overview?.molecule_labels)) {
        overview.molecule_labels.forEach((label: any, idx: number) => {
            labelByMoleculeId[String(idx)] = String(label);
        });
    }

    const splitById = new Map<string, SplitTag>();
    const splitNames: string[] = Array.isArray(overview?.split_names) ? overview.split_names.map((x: any) => String(x).toLowerCase()) : [];
    const splits: any[] = Array.isArray(overview?.splits) ? overview.splits : [];
    const selectedSplitSet = Array.isArray(selectedSplits) && selectedSplits.length > 0
        ? new Set(selectedSplits.map((x) => String(x).toLowerCase()))
        : null;
    const allowedMolIds = new Set<string>();
    if (selectedSplitSet) {
        splitNames.forEach((name, idx) => {
            if (!selectedSplitSet.has(name)) return;
            const ids: any[] = Array.isArray(splits[idx]) ? splits[idx] : [];
            ids.forEach((id) => allowedMolIds.add(String(id)));
        });
    }
    splitNames.forEach((name, idx) => {
        const ids: any[] = Array.isArray(splits[idx]) ? splits[idx] : [];
        const split: SplitTag | null =
            name === "train" ? "train" : name === "test" ? "test" : name === "val" ? "val" : null;
        if (!split) return;
        ids.forEach((id) => {
            splitById.set(String(id), split);
        });
    });

    const rows: SampleRow[] = [];
    const addPair = (
        fromEpochIdx: number,
        fromLayerName: string,
        toEpochIdx: number,
        toLayerName: string
    ) => {
        const fromLayerIdx = layerIndexByName.get(fromLayerName);
        const toLayerIdx = layerIndexByName.get(toLayerName);
        if (fromLayerIdx == null || toLayerIdx == null) return;
        const fromCellIdx = fromLayerIdx + fromEpochIdx * layersRaw.length;
        const toCellIdx = toLayerIdx + toEpochIdx * layersRaw.length;
        const fromLabels = cells[fromCellIdx]?.cluster_labels;
        const toLabels = cells[toCellIdx]?.cluster_labels;
        if (!fromLabels || !toLabels) return;
        Object.entries(fromLabels).forEach(([molIdRaw, fromClusterRaw]) => {
            if (selectedSplitSet && !allowedMolIds.has(String(molIdRaw))) return;
            const toClusterRaw = (toLabels as Record<string, unknown>)[molIdRaw];
            if (toClusterRaw == null) return;
            const fromCluster = Number(fromClusterRaw);
            const toCluster = Number(toClusterRaw);
            if (!Number.isFinite(fromCluster) || !Number.isFinite(toCluster)) return;
            const molId = String(molIdRaw);
            const visibleId = labelByMoleculeId[molId] ?? molId;
            const split = splitById.get(molId);
            // Keep val ids tagged as test for downstream id lookups; split stays on row.split for rendering.
            const taggedSplit = split ? (split === "train" ? "train" : "test") : null;
            const realId = taggedSplit ? `${taggedSplit}::${visibleId}` : visibleId;
            rows.push({
                real_id: realId,
                from_model: epochs[fromEpochIdx],
                to_model: epochs[toEpochIdx],
                from_layer: fromLayerName,
                to_layer: toLayerName,
                from_cluster: fromCluster,
                to_cluster: toCluster,
                split,
            });
        });
    };

    // Contextual rows:
    // - mode=model  => transitions across epochs within selected layer
    // - mode=layer  => transitions across layers within selected epoch(model)
    if (mode === "model") {
        const wantedLayer = canonicalLayerKey(String(layerKey ?? ""));
        const chosenLayer =
            layers.find((name) => canonicalLayerKey(name) === wantedLayer) ??
            layers.find((name) => canonicalLayerKey(name) === canonicalLayerKey(normalizeLayerForVisibility(layerKey)));
        if (chosenLayer) {
            for (let e = 0; e < epochs.length - 1; e += 1) {
                addPair(e, chosenLayer, e + 1, chosenLayer);
            }
            return rows;
        }
    }

    if (mode === "layer") {
        const wantedModel = String(modelKey ?? "");
        const epochIdx = epochs.findIndex((e) => String(e) === wantedModel);
        if (epochIdx >= 0) {
            for (let l = 0; l < layers.length - 1; l += 1) {
                addPair(epochIdx, layers[l], epochIdx, layers[l + 1]);
            }
            return rows;
        }
    }

    // Fallback: full row generation.
    for (let e = 0; e < epochs.length; e += 1) {
        for (let l = 0; l < layers.length - 1; l += 1) {
            addPair(e, layers[l], e, layers[l + 1]);
        }
    }
    for (let e = 0; e < epochs.length - 1; e += 1) {
        for (let l = 0; l < layers.length; l += 1) {
            addPair(e, layers[l], e + 1, layers[l]);
        }
    }

    return rows;
}

type SplitTag = "train" | "test" | "val";
const HIDDEN_LAYER = "ffn_layer_2";

function normalizeLayerForVisibility(name: unknown): string {
    const s = String(name ?? "").trim();
    if (s === "fp") return "fingerprint";
    if (/^fnn_layer_/.test(s)) return s.replace(/^fnn_layer_/, "ffn_layer_");
    return s;
}

function isVisibleRow(row: SampleRow): boolean {
    const fromLayer = normalizeLayerForVisibility(row.from_layer);
    const toLayer = normalizeLayerForVisibility(row.to_layer);
    return fromLayer !== HIDDEN_LAYER && toLayer !== HIDDEN_LAYER;
}

function inferSplitFromPath(path: string | undefined): SplitTag {
    const s = String(path ?? "").toLowerCase();
    if (s.includes("split=val")) return "val";
    return s.includes("split=test") ? "test" : "train";
}

function stripTaggedRealId(value: string | number): { id: string; split: SplitTag | null } {
    const raw = String(value ?? "");
    if (raw.startsWith("train::")) return { id: raw.slice("train::".length), split: "train" };
    if (raw.startsWith("test::")) return { id: raw.slice("test::".length), split: "test" };
    if (raw.startsWith("val::")) return { id: raw.slice("val::".length), split: "val" };
    return { id: raw, split: null };
}

function ensureTaggedSampleId(value: string | number, split: SplitTag): string {
    const parsed = stripTaggedRealId(value);
    if (parsed.split) return String(value);
    const splitForId = split === "train" ? "train" : "test";
    return `${splitForId}::${parsed.id}`;
}

function stitchRowsByVisibleEpochs(rows: SampleRow[], visibleEpochs: string[]): SampleRow[] {
    if (!Array.isArray(visibleEpochs) || visibleEpochs.length < 2) return [];
    const orderedEpochs = visibleEpochs.map((x) => String(x));
    const allowed = new Set(orderedEpochs);
    const assignmentById = new Map<
        string,
        Map<string, { cluster: number; model?: string; layer?: string; split?: SplitTag }>
    >();

    rows.forEach((r) => {
        const fromModel = String(r.from_model ?? "");
        const toModel = String(r.to_model ?? "");
        const id = String(r.real_id);
        if (!id) return;
        if (!assignmentById.has(id)) assignmentById.set(id, new Map());
        const amap = assignmentById.get(id)!;
        const parsedSplit = stripTaggedRealId(id).split;
        const rowSplit = r.split ?? parsedSplit ?? undefined;
        if (allowed.has(fromModel)) {
            amap.set(fromModel, { cluster: r.from_cluster, model: r.from_model, layer: r.from_layer, split: rowSplit });
        }
        if (allowed.has(toModel)) {
            amap.set(toModel, { cluster: r.to_cluster, model: r.to_model, layer: r.to_layer, split: rowSplit });
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
                split: fromAssign.split ?? toAssign.split,
            });
        }
    });
    return out;
}

type SankeyProps = {
    data: any;
    selectedDataSet: string;
    selectedSplits?: string[];
    secondaryDataPath?: string;
    showFfn2?: boolean;
    syncTransform?: { x: number; y: number; scale: number };
    onSyncTransform?: (t: { x: number; y: number; scale: number }) => void;
    mode?: "layer" | "model";
    layerKey?: string;
    modelKey?: string;
    visibleEpochs?: string[] | null;
    reverseEpochOrder?: boolean;
    activeNodeKey?: string | null;
    onNodeSelect?: (sel: { model?: string; layer?: string; cluster: number; ids?: string[] }) => void;
    layout: "vertical" | "horizontal"
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
        const fromRaw = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer ?? r.from_model;
        const toRaw = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer ?? r.to_model;
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
    if (mode !== "layer") {
        if (fromPreferred.length > 0) {
            const seen = new Set(fromPreferred);
            const remaining = fallback.filter((k) => !seen.has(k));
            return [...fromPreferred, ...remaining];
        }
        return fallback;
    }

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
        Map<string, { cluster: number; model?: string; layer?: string; split?: SplitTag }>
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
        const parsedSplit = stripTaggedRealId(rid).split;
        const rowSplit = r.split ?? parsedSplit ?? undefined;
        amap.set(fk, { cluster: r.from_cluster, model: r.from_model, layer: r.from_layer, split: rowSplit });
        amap.set(tk, { cluster: r.to_cluster, model: r.to_model, layer: r.to_layer, split: rowSplit });
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
                split: fromAssign.split ?? toAssign.split,
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
const MAX_LAYER_GAP = 90;
const MAX_EPOCH_GAP = 140;
const MIN_LAYER_GAP_HORIZONTAL = 30;
const MIN_EPOCH_GAP_HORIZONTAL = 60;
const MAX_LAYER_GAP_HORIZONTAL = 320;
const MAX_EPOCH_GAP_HORIZONTAL = 360;
const SYNC_VERTICAL_SHIFT_RATIO = 0.5;
const CELL_W = 120;
const CELL_MARGIN_LEFT = 10;
const NODE_W = 12;
const NODE_PAD = 4;
const BOTTOM = 28;
const TOP = 10;
const INNER_W = 100;
const labelYOffset = -8;

const UNSYNC_SHIFT_X = 0;
const UNSYNC_LEFT_PAD = 20;
const UNSYNC_RIGHT_PAD = 20;
const centerOffset = CELL_MARGIN_LEFT + INNER_W / 2;


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

export default function Sankey({
    data,
    selectedDataSet,
    selectedSplits,
    secondaryDataPath,
    showFfn2 = false,
    syncTransform,
    onSyncTransform,
    mode = "model",
    layerKey,
    modelKey,
    visibleEpochs,
    reverseEpochOrder = false,
    activeNodeKey,
    onNodeSelect,
    layout = "vertical",
}: SankeyProps) {
    const containerRef = useRef<HTMLDivElement | null>(null);
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
        if (!container || !data) return;
        const host = container;

        const groupsRef = {
            links: null as SVGGElement | null,
            sampleTrajectories: null as SVGGElement | null,
            nodes: null as SVGGElement | null,
            axisLabels: null as SVGGElement | null,
        };
        const pendingFrame = { id: 0 as number | null };
        const lastRowsRef = { current: null as SampleRow[] | null };
        const lastSecondaryRowsRef = { current: null as SampleRow[] | null };

        let rotationAxis: SVGElement | null = null;

        let svg = d3.select(container).select<SVGSVGElement>("svg.app-sankey");
        if (svg.empty()) {
            svg = d3
                .select(container)
                .append("svg")
                .attr("class", "app-sankey")
                .style("width", "100%")
                .style("height", "100%");

            // seems like this is the only thing that sets the rotation
            rotationAxis = svg.append("g").node();
            if (layout === "vertical") {
                d3.select(rotationAxis).attr("transform", `translate(${container.clientWidth},0) rotate(90)`).node();
            }
            groupsRef.links = d3.select(rotationAxis).append("g").attr("class", "links").node() as SVGGElement;
            groupsRef.sampleTrajectories = d3.select(rotationAxis).append("g").attr("class", "sample-trajectories").node() as SVGGElement;
            groupsRef.nodes = d3.select(rotationAxis).append("g").attr("class", "nodes").node() as SVGGElement;
            groupsRef.axisLabels = svg.append("g").attr("class", "axis-labels").node() as SVGGElement;
        }

        const ro = new ResizeObserver(() => draw());
        ro.observe(container);

        let lastRows: SampleRow[] | null = null;
        let lastSecondaryRows: SampleRow[] | null = null;

        // do the comparison once
        const fromKeyFn = mode === "model" ? (d) => {
            return d.from_model;
        } : (d) => { return d.from_layer };

        const toKeyFn = mode === "model" ? (d) => {
            return d.to_model;
        } : (d) => { return d.to_layer };

        const rowsFromData = (() => {
            if (Array.isArray(data)) return data as SampleRow[];
            return rowsFromCompactOverview(data, showFfn2, selectedSplits, mode, modelKey, layerKey);
        })();
        lastRows = Array.isArray(rowsFromData) ? rowsFromData.filter((r) => showFfn2 || isVisibleRow(r)) : [];
        lastSecondaryRows = null;

        function draw() {
            if (!container) return;
            const baseRows = Array.isArray(lastRows) ? lastRows : [];

            const width = Math.max(1, container.clientWidth);
            const height = Math.max(1, container.clientHeight);
            let drawW = width;
            let drawH = height;

            svg.attr("width", width).attr("height", height);
            if (layout === "vertical") {
                drawW = height;
                drawH = width;
                d3.select(rotationAxis).attr("transform", `translate(${width},0) rotate(90)`);
            }

            const axisLabels = d3.select(groupsRef.axisLabels!);
            // model transition within a single layer
            const epochsFromData =
                Array.isArray(data?.epochs) ? data.epochs.map((x: any) => String(x)) : [];
            const visibleEpochOrder =
                mode === "model" && Array.isArray(visibleEpochs) && visibleEpochs.length > 0
                    ? (reverseEpochOrder ? [...visibleEpochs].reverse() : visibleEpochs).map((x) => String(x))
                    : null;
            const renderOrder =
                mode === "model"
                    ? (visibleEpochOrder && visibleEpochOrder.length > 0
                        ? visibleEpochOrder
                        : (reverseEpochOrder ? [...epochsFromData].reverse() : epochsFromData))
                    : (Array.isArray(data?.layers) ? data.layers.map((x: any) => String(x)) : []);
            axisLabels.selectAll("text").remove();

            const axisPadStart = layout === "vertical" ? UNSYNC_TOP_PAD : UNSYNC_LEFT_PAD;
            const axisPadEnd = layout === "vertical" ? UNSYNC_BOTTOM_PAD : UNSYNC_RIGHT_PAD;
            const minGap =
                mode === "model"
                    ? (layout === "horizontal" ? MIN_EPOCH_GAP_HORIZONTAL : MIN_EPOCH_GAP)
                    : (layout === "horizontal" ? MIN_LAYER_GAP_HORIZONTAL : MIN_LAYER_GAP);
            const maxGap =
                mode === "model"
                    ? (layout === "horizontal" ? MAX_EPOCH_GAP_HORIZONTAL : MAX_EPOCH_GAP)
                    : (layout === "horizontal" ? MAX_LAYER_GAP_HORIZONTAL : MAX_LAYER_GAP);
            const slotCount = Math.max(1, renderOrder.length);
            const usableAxisSpan = Math.max(1, drawW - axisPadStart - axisPadEnd);
            let axisGap = 0;
            if (slotCount > 1) {
                const rawGap = (usableAxisSpan - NODE_W) / (slotCount - 1);
                axisGap = Math.max(minGap, Math.min(maxGap, rawGap));
                // If min-gap cannot fit, fall back to the available span to avoid clipping.
                if (NODE_W + axisGap * (slotCount - 1) > usableAxisSpan) {
                    axisGap = Math.max(0, rawGap);
                }
            }
            const occupiedAxisSpan = NODE_W + axisGap * Math.max(0, slotCount - 1);
            const axisStart = axisPadStart + Math.max(0, (usableAxisSpan - occupiedAxisSpan) / 2);
            const axisCenterFor = (idx: number) => axisStart + idx * axisGap + NODE_W / 2;

            if (layout === "vertical") {
                axisLabels
                    .selectAll("text")
                    .data(renderOrder)
                    .join("text")
                    .attr("x", width - 6)
                    .attr("y", (_d, i) => axisCenterFor(i) + labelYOffset)
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
            } else {
                axisLabels
                    .selectAll("text")
                    .data(renderOrder)
                    .join("text")
                    .attr("x", (_d, i) => axisCenterFor(i))
                    .attr("y", height - 14)
                    .attr("text-anchor", "middle")
                    .attr("dominant-baseline", "middle")
                    .attr("font-size", 12)
                    .attr("font-weight", 600)
                    .attr("fill", "#374151")
                    .attr("stroke", "#ffffff")
                    .attr("stroke-width", 2)
                    .attr("paint-order", "stroke")
                    .style("pointer-events", "none")
                    .text((d) => toShortLayerName(String(d)));

            }
            const order = mode === "model" ? renderOrder : LAYER_ORDER;
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
            const filteredRows = filterRowsForView(baseRows);
            const filteredSecondaryRows = lastSecondaryRows ? filterRowsForView(lastSecondaryRows) : [];

            const rowsReal = normalizeAdjacentRows(filteredRows, mode, order) as any[];
            const rowsSynthetic = synthesizeMissingAdjacentRows(rowsReal, mode, order) as any[];
            const rows = rowsSynthetic.length > 0 ? [...rowsReal, ...rowsSynthetic] : rowsReal;
            const secondaryRowsReal = normalizeAdjacentRows(filteredSecondaryRows, mode, order) as any[];
            const secondaryRowsSynthetic = synthesizeMissingAdjacentRows(secondaryRowsReal, mode, order) as any[];
            const secondaryRows =
                secondaryRowsSynthetic.length > 0 ? [...secondaryRowsReal, ...secondaryRowsSynthetic] : secondaryRowsReal;
            const combinedRows = secondaryRows.length > 0 ? [...rows, ...secondaryRows] : rows;
            const MARGIN_LEFT = CELL_H * 2 + 10;
            const bandH = Math.max(1, drawH - TOP - BOTTOM);

            const nodeValue = new Map<string, number>();
            combinedRows.forEach((r) => {
                const fromKey = fromKeyFn(r);
                const toKey = toKeyFn(r);
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
            const inferredOrderCombined = inferSequenceFromRows(combinedRows as SampleRow[], mode, order);

            const axisEnd = axisStart + occupiedAxisSpan;
            // In vertical mode the rotated x-axis maps to screen Y; allow overflow with native scroll if needed.
            const totalHeight = layout === "vertical" ? Math.max(height, Math.ceil(axisEnd + 18)) : height;
            svg.attr("height", totalHeight);

            // Layer/model labels for each row (right side; screen-space, not rotated with data group).


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
                    const x = axisCenterFor(i) - NODE_W / 2;
                    nodes.push({ key: n.key, model, value: n.value, x, y: cur, h });
                    cur += h + NODE_PAD;
                });
            });
            //console.log(layout);
            //console.log(nodes);

            let activeModel: string | null = null;
            if (displayActiveNode) {
                const idx = displayActiveNode.lastIndexOf("_");
                activeModel = idx === -1 ? displayActiveNode : displayActiveNode.slice(0, idx);
            }
            const focusGroup = activeModel ?? (mode === "model" ? modelKey ?? null : layerKey ?? null);
            const primarySplit: SplitTag =
                String(selectedDataSet ?? "").toLowerCase() === "dimenet" ? "test" : "train";
            const secondarySplit: SplitTag = secondaryDataPath
                ? inferSplitFromPath(String(secondaryDataPath))
                : primarySplit;
            const primaryRowsWithSplit = rows.map((r) => ({ ...r, __split: primarySplit as SplitTag }));
            const secondaryRowsWithSplit = secondaryRows.map((r) => ({ ...r, __split: secondarySplit as SplitTag }));
            const combinedRowsWithSplit =
                secondaryRowsWithSplit.length > 0 ? [...primaryRowsWithSplit, ...secondaryRowsWithSplit] : primaryRowsWithSplit;
            const splitCountByNode = new Map<string, { train: number; test: number; val: number }>();
            const normalizeSplit = (value: unknown): SplitTag | null => {
                if (value === "train" || value === "test" || value === "val") return value;
                return null;
            };
            const getRowSplit = (r: any): SplitTag => {
                const rowSplit = normalizeSplit(r?.split);
                if (rowSplit) return rowSplit;
                const parsedSplit = stripTaggedRealId(r?.real_id ?? "").split;
                if (parsedSplit) return parsedSplit;
                const fallback = normalizeSplit(r?.__split);
                return fallback ?? primarySplit;
            };
            const addSplitCount = (nodeKey: string, split: SplitTag) => {
                const counts = splitCountByNode.get(nodeKey) ?? { train: 0, test: 0, val: 0 };
                if (split === "train") counts.train += 1;
                else if (split === "test") counts.test += 1;
                else counts.val += 1;
                splitCountByNode.set(nodeKey, counts);
            };
            combinedRowsWithSplit.forEach((r: any) => {
                const fromKey = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer ?? r.from_model;
                const toKey = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer ?? r.to_model;
                if (!fromKey || !toKey) return;
                const src = `${fromKey}_${r.from_cluster}`;
                const tgt = `${toKey}_${r.to_cluster}`;
                const split = getRowSplit(r);
                addSplitCount(src, split);
                addSplitCount(tgt, split);
            });

            const clusterValues: number[] = [];
            nodes.forEach((n) => {
                if (!focusGroup || n.model !== focusGroup) return;
                const idx = n.key.lastIndexOf("_");
                const cluster = Number(idx === -1 ? 0 : n.key.slice(idx + 1));
                if (Number.isFinite(cluster)) clusterValues.push(cluster);
            });
            const maxCluster = clusterValues.length ? Math.max(...clusterValues) : 0;

            const color = d3
                .scaleOrdinal<number, string>()
                .domain(d3.range(maxCluster + 1))
                .range(CLUSTER_COLORS);

            const nodeByKey = new Map(nodes.map((n) => [n.key, n]));
            const linkMap = new Map<string, number>();
            combinedRows.forEach((r) => {
                const fromKey = fromKeyFn(r);
                const toKey = toKeyFn(r);
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

            const linkGroup = d3.select(groupsRef.links!);
            const sampleGroup = d3.select(groupsRef.sampleTrajectories!);
            const nodeGroup = d3.select(groupsRef.nodes!);

            linkGroup
                .selectAll<SVGPathElement, { src: string; tgt: string; value: number }>("path")
                .data(links, (d: any) => `${d.src}->${d.tgt}`)
                .join(
                    (enter) =>
                        enter.append("path")
                            .attr("fill", "none")
                            .attr("stroke", SANKEY_LINK_COLOR)
                            .attr("stroke-opacity", 0.35)
                            .attr("vector-effect", "non-scaling-stroke"),
                    (update) => update,
                    (exit) => exit.remove()
                )
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
                .attr("stroke-width", (d) => {
                    const s = nodeByKey.get(d.src);
                    if (!s) return 1;
                    const sTot = outTotal.get(d.src) || 1;
                    return Math.max(0.5, (d.value / sTot) * s.h);
                });

            const nodeSamples = new Map<string, Set<string>>();
            combinedRowsWithSplit.forEach((r: any) => {
                const fromKey = mode === "model" ? r.from_model ?? r.from_layer : r.from_layer ?? r.from_model;
                const toKey = mode === "model" ? r.to_model ?? r.to_layer : r.to_layer ?? r.to_model;
                const src = `${fromKey}_${r.from_cluster}`;
                const tgt = `${toKey}_${r.to_cluster}`;
                const parsed = stripTaggedRealId(r.real_id);
                const split: SplitTag = (parsed.split ?? r.__split ?? primarySplit) as SplitTag;
                const sampleId = ensureTaggedSampleId(r.real_id, split);
                if (!nodeSamples.has(src)) nodeSamples.set(src, new Set());
                if (!nodeSamples.has(tgt)) nodeSamples.set(tgt, new Set());
                nodeSamples.get(src)!.add(sampleId);
                nodeSamples.get(tgt)!.add(sampleId);
            });
            const darker = (hex: string, amount = 0.6) => {
                const c = d3.color(hex);
                return c ? c.darker(amount).formatHex() : hex;
            };

            const nodeSel = nodeGroup
                .selectAll("rect.node-main")
                .data(nodes, (d: any) => d.key);

            nodeSel.join(
                (enter) =>
                    enter.append("rect")
                        .attr("class", "node-main")
                        .attr("x", (d) => d.x)
                        .attr("y", (d) => d.y)
                        .attr("width", NODE_W)
                        .attr("height", (d) => Math.max(1, d.h))
                        .style("cursor", "pointer")
                        .attr("data-key", (d) => d.key)
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
                        })
                        .attr("fill", (d) => {
                            if (focusGroup && d.model === focusGroup) {
                                const idx = d.key.lastIndexOf("_");
                                const cluster = Number(idx === -1 ? 0 : d.key.slice(idx + 1));
                                if (Number.isFinite(cluster)) return color(Math.max(0, cluster));
                            }
                            return SANKEY_NODE_COLOR;
                        })
                        .attr("stroke", "none")
                        .attr("stroke-width", 0),
                (update) => update
                    .attr("x", (d: any) => d.x)
                    .attr("y", (d) => d.y)
                    .attr("height", (d) => Math.max(1, d.h))
                    .attr("fill", (d) => {
                        if (focusGroup && d.model === focusGroup) {
                            const idx = d.key.lastIndexOf("_");
                            const cluster = Number(idx === -1 ? 0 : d.key.slice(idx + 1));
                                if (Number.isFinite(cluster)) return color(Math.max(0, cluster));
                            }
                            return SANKEY_NODE_COLOR;
                        })
                    .attr("stroke", "none")
                    .attr("stroke-width", 0),
                (exit) => exit.remove());

            nodeGroup
                .selectAll("rect.node-test")
                .data(nodes).join(
                    (enter) =>
                        enter
                            .append("rect")
                            .attr("class", "node-test")
                            .attr("x", (d) => d.x)
                            .attr("y", (d) => {
                                const total = Math.max(0, Number(d.value ?? 0));
                                const splitCounts = splitCountByNode.get(d.key) ?? { train: 0, test: 0, val: 0 };
                                const test = Math.max(0, Number(splitCounts.test));
                                const val = Math.max(0, Number(splitCounts.val));
                                const ratio = total > 0 ? Math.min(1, test / total) : 0;
                                const valRatio = total > 0 ? Math.min(1, val / total) : 0;
                                const h = Math.max(1, d.h);
                                const valH = h * valRatio;
                                const testH = h * ratio;
                                return d.y + Math.max(0, h - valH - testH);
                            })
                            .attr("width", NODE_W)
                            .attr("height", (d) => {
                                const total = Math.max(0, Number(d.value ?? 0));
                                const splitCounts = splitCountByNode.get(d.key) ?? { train: 0, test: 0, val: 0 };
                                const test = Math.max(0, Number(splitCounts.test));
                                const ratio = total > 0 ? Math.min(1, test / total) : 0;
                                return Math.max(0, Math.max(1, d.h) * ratio);
                            })
                            .attr("fill", (d) => {
                                if (focusGroup && d.model === focusGroup) {
                                    const idx = d.key.lastIndexOf("_");
                                    const cluster = Number(idx === -1 ? 0 : d.key.slice(idx + 1));
                                    if (Number.isFinite(cluster)) return darker(color(Math.max(0, cluster)), 0.6);
                                }
                                return darker(SANKEY_NODE_COLOR, 0.6);
                            })
                            .attr("data-key", (d) => d.key)
                            .attr("stroke", "none")
                            .attr("stroke-width", 0)
                            .style("pointer-events", "none"),
                    (update) =>
                        update
                            .attr("x", (d) => d.x)
                            .attr("y", (d) => {
                                const total = Math.max(0, Number(d.value ?? 0));
                                const splitCounts = splitCountByNode.get(d.key) ?? { train: 0, test: 0, val: 0 };
                                const test = Math.max(0, Number(splitCounts.test));
                                const val = Math.max(0, Number(splitCounts.val));
                                const ratio = total > 0 ? Math.min(1, test / total) : 0;
                                const valRatio = total > 0 ? Math.min(1, val / total) : 0;
                                const h = Math.max(1, d.h);
                                const valH = h * valRatio;
                                const testH = h * ratio;
                                return d.y + Math.max(0, h - valH - testH);
                            }).attr("height", (d) => {
                                const total = Math.max(0, Number(d.value ?? 0));
                                const splitCounts = splitCountByNode.get(d.key) ?? { train: 0, test: 0, val: 0 };
                                const test = Math.max(0, Number(splitCounts.test));
                                const ratio = total > 0 ? Math.min(1, test / total) : 0;
                                return Math.max(0, Math.max(1, d.h) * ratio);
                            })
                            .attr("fill", (d) => {
                                if (focusGroup && d.model === focusGroup) {
                                    const idx = d.key.lastIndexOf("_");
                                    const cluster = Number(idx === -1 ? 0 : d.key.slice(idx + 1));
                                    if (Number.isFinite(cluster)) return darker(color(Math.max(0, cluster)), 0.6);
                                }
                                return darker(SANKEY_NODE_COLOR, 0.6);
                            })
                            .attr("data-key", (d) => d.key)
                            .attr("stroke", "none")
                            .attr("stroke-width", 0)
                    ,
                    (exit) => exit.remove()
                );
            nodeGroup
                .selectAll("rect.node-val")
                .data(nodes).join(
                    (enter) =>
                        enter
                            .append("rect")
                            .attr("class", "node-val")
                            .attr("x", (d) => d.x)
                            .attr("y", (d) => {
                                const total = Math.max(0, Number(d.value ?? 0));
                                const splitCounts = splitCountByNode.get(d.key) ?? { train: 0, test: 0, val: 0 };
                                const val = Math.max(0, Number(splitCounts.val));
                                const ratio = total > 0 ? Math.min(1, val / total) : 0;
                                const h = Math.max(1, d.h);
                                const valH = h * ratio;
                                return d.y + Math.max(0, h - valH);
                            })
                            .attr("width", NODE_W)
                            .attr("height", (d) => {
                                const total = Math.max(0, Number(d.value ?? 0));
                                const splitCounts = splitCountByNode.get(d.key) ?? { train: 0, test: 0, val: 0 };
                                const val = Math.max(0, Number(splitCounts.val));
                                const ratio = total > 0 ? Math.min(1, val / total) : 0;
                                return Math.max(0, Math.max(1, d.h) * ratio);
                            })
                            .attr("fill", (d) => {
                                if (focusGroup && d.model === focusGroup) {
                                    const idx = d.key.lastIndexOf("_");
                                    const cluster = Number(idx === -1 ? 0 : d.key.slice(idx + 1));
                                    if (Number.isFinite(cluster)) return darker(color(Math.max(0, cluster)), 1.0);
                                }
                                return darker(SANKEY_NODE_COLOR, 1.0);
                            })
                            .attr("data-key", (d) => d.key)
                            .attr("stroke", "none")
                            .attr("stroke-width", 0)
                            .style("pointer-events", "none"),
                    (update) =>
                        update
                            .attr("x", (d) => d.x)
                            .attr("y", (d) => {
                                const total = Math.max(0, Number(d.value ?? 0));
                                const splitCounts = splitCountByNode.get(d.key) ?? { train: 0, test: 0, val: 0 };
                                const val = Math.max(0, Number(splitCounts.val));
                                const ratio = total > 0 ? Math.min(1, val / total) : 0;
                                const h = Math.max(1, d.h);
                                const valH = h * ratio;
                                return d.y + Math.max(0, h - valH);
                            })
                            .attr("height", (d) => {
                                const total = Math.max(0, Number(d.value ?? 0));
                                const splitCounts = splitCountByNode.get(d.key) ?? { train: 0, test: 0, val: 0 };
                                const val = Math.max(0, Number(splitCounts.val));
                                const ratio = total > 0 ? Math.min(1, val / total) : 0;
                                return Math.max(0, Math.max(1, d.h) * ratio);
                            })
                            .attr("fill", (d) => {
                                if (focusGroup && d.model === focusGroup) {
                                    const idx = d.key.lastIndexOf("_");
                                    const cluster = Number(idx === -1 ? 0 : d.key.slice(idx + 1));
                                    if (Number.isFinite(cluster)) return darker(color(Math.max(0, cluster)), 1.0);
                                }
                                return darker(SANKEY_NODE_COLOR, 1.0);
                            })
                            .attr("data-key", (d) => d.key)
                            ,
                    (exit) => exit.remove()
                );
            nodeGroup
                .selectAll("rect.node-outline")
                .data(nodes, (d: any) => d.key)
                .join(
                    (enter) =>
                        enter
                            .append("rect")
                            .attr("class", "node-outline")
                            .attr("fill", "none")
                            .attr("data-key", (d) => d.key)
                            .style("pointer-events", "none"),
                    (update) => update,
                    (exit) => exit.remove()
                )
                .attr("x", (d) => d.x)
                .attr("y", (d) => d.y)
                .attr("width", NODE_W)
                .attr("height", (d) => Math.max(1, d.h))
                .attr("stroke", (d) => {
                    if (displayActiveNode && d.key === displayActiveNode) return HIGHLIGHT_RED;
                    return "none";
                })
                .attr("stroke-width", (d) => {
                    if (displayActiveNode && d.key === displayActiveNode) return 2.6;
                    return 0;
                })
                .attr("vector-effect", "non-scaling-stroke");
            nodeSel.append("title").text((d) => d.key);
            nodeGroup.raise();

            // Keep labels above links/nodes.
            svg.selectAll("g.axis-labels").raise();

            // Re-apply current transform right after redraw so viewport state is preserved.
            const contentGroups = svg.selectAll("g.links, g.sample-trajectories, g.nodes");
            const labelGroups = svg.selectAll("g.axis-labels");
            if (syncTransform) {
                const scale = syncTransform.scale;
                const offsetX = syncTransform.y - SYNC_BASE_OFFSET * scale;
                contentGroups.attr("transform", `translate(${offsetX},0) scale(${scale},1)`);
                if (layout === "horizontal") {
                    labelGroups.attr("transform", `translate(${offsetX},0) scale(${scale},1)`);
                } else {
                    labelGroups.attr("transform", `translate(0,${offsetX}) scale(1,${scale})`);
                }
            } else if (panY !== 0 || zoomX !== 1) {
                contentGroups.attr("transform", `translate(${panY},0) scale(${zoomX},1)`);
                if (layout === "horizontal") {
                    labelGroups.attr("transform", `translate(${panY},0) scale(${zoomX},1)`);
                } else {
                    labelGroups.attr("transform", `translate(0,${panY}) scale(1,${zoomX})`);
                }
            } else {
                contentGroups.attr("transform", null);
                labelGroups.attr("transform", null);
            }
        }

        /*Promise.all([
            fetchJsonCached(dataPath),
            secondaryDataPath
                ? fetchJsonCached(secondaryDataPath)
                    .then((json) => json ?? null)
                    .catch(() => null)
                : Promise.resolve(null),
        ])
            .then(([json, secondaryJson]: [SampleRow[], SampleRow[] | null]) => {
                const pickRows = (rows: SampleRow[] | null) =>
                    Array.isArray(rows) ? (showFfn2 ? rows : rows.filter(isVisibleRow)) : null;
                lastRows = pickRows(json) ?? [];
                lastSecondaryRows = pickRows(secondaryJson);
                draw();
            })
            .catch((err) => {
                console.error(err);
                if (container) container.innerHTML = "<div style='color:red'>Failed to load sankey data.</div>";
            });*/

        draw();

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
    }, [
        data,
        selectedDataSet,
        selectedSplits,
        mode,
        modelKey,
        layerKey,
        visibleEpochs,
        reverseEpochOrder,
        showFfn2,
        !!syncTransform,
        activeNodeKey,
    ]);

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;
        const svg = container.querySelector("svg");
        if (!svg) return;

        const activeName = displayActiveNode;
        const activeModel = activeName ? activeName.slice(0, Math.max(0, activeName.lastIndexOf("_"))) : null;
        const focusGroup = activeModel ?? (mode === "model" ? modelKey ?? null : layerKey ?? null);

        const fillRects = Array.from(
            svg.querySelectorAll("g.nodes rect.node-main[data-key], g.nodes rect.node-test[data-key], g.nodes rect.node-val[data-key]")
        ) as SVGRectElement[];
        fillRects.forEach((rect) => {
            const key = rect.getAttribute("data-key") || "";
            const idx = key.lastIndexOf("_");
            const model = idx === -1 ? key : key.slice(0, idx);
            const cluster = Number(idx === -1 ? 0 : key.slice(idx + 1));
            const isTestRect = rect.classList.contains("node-test");
            const isValRect = rect.classList.contains("node-val");
            const baseColor =
                focusGroup && model === focusGroup && Number.isFinite(cluster)
                    ? CLUSTER_COLORS[Math.max(0, cluster) % CLUSTER_COLORS.length]
                    : SANKEY_NODE_COLOR;
            const c = d3.color(baseColor);
            const testColor = c ? c.darker(0.6).formatHex() : baseColor;
            const valColor = c ? c.darker(1.0).formatHex() : baseColor;

            rect.setAttribute("fill", isValRect ? valColor : isTestRect ? testColor : baseColor);

            rect.setAttribute("stroke", "none");
            rect.setAttribute("stroke-width", "0");
        });

        const outlineRects = Array.from(
            svg.querySelectorAll("g.nodes rect.node-outline[data-key]")
        ) as SVGRectElement[];
        outlineRects.forEach((rect) => {
            const key = rect.getAttribute("data-key") || "";
            if (activeName && key === activeName) {
                rect.setAttribute("stroke", HIGHLIGHT_RED);
                rect.setAttribute("stroke-width", "2.6");
            } else {
                rect.setAttribute("stroke", "none");
                rect.setAttribute("stroke-width", "0");
            }
        });

    }, [displayActiveNode, mode, modelKey, layerKey]);

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
            labelTransformAttr =
                layout === "horizontal"
                    ? `translate(${offsetX},0) scale(${scale},1)`
                    : `translate(0,${offsetX}) scale(1,${scale})`;
        } else if (panY !== 0 || zoomX !== 1) {
            // In rotated space, x-translate maps to on-screen vertical movement.
            contentTransformAttr = `translate(${panY},0) scale(${zoomX},1)`;
            labelTransformAttr =
                layout === "horizontal"
                    ? `translate(${panY},0) scale(${zoomX},1)`
                    : `translate(0,${panY}) scale(1,${zoomX})`;
        }

        contentGroups.forEach((el) => {
            if (contentTransformAttr) (el as SVGGElement).setAttribute("transform", contentTransformAttr);
            else (el as SVGGElement).removeAttribute("transform");
        });
        labelGroups.forEach((el) => {
            if (labelTransformAttr) (el as SVGGElement).setAttribute("transform", labelTransformAttr);
            else (el as SVGGElement).removeAttribute("transform");
        });
    }, [syncTransform, panY, zoomX, layout]);

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
