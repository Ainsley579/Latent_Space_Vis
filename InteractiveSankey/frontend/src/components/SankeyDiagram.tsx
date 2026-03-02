"use client";
import * as d3 from "d3";
import {
    sankey,
    sankeyLinkHorizontal,
    SankeyNodeMinimal,
    SankeyLinkMinimal,
} from "d3-sankey";
import React, { useEffect, useRef, useState, useImperativeHandle } from "react";

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
interface SankeyDiagramProps {
    data?: {
        real_id: string | number;
        from_layer: string;
        to_layer: string;
        from_cluster: number;
        to_cluster: number;
        split_label?: "train" | "test" | "experimental" | "missing";
    }[];
    sampleFile?: string;
    hideTooltip?: boolean;
    zoomFactor?: number;
}
type Split = "train" | "test" | "experimental" | "missing";


interface LineDatum {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    stroke: string;
    className: string;
}

const MAX_TRAJECTORIES_TOTAL = 18000;
const MAX_TRAJECTORIES_PER_LINK = 500;

function strideSample<T>(arr: T[], maxCount: number): T[] {
    if (arr.length <= maxCount) return arr;
    const step = Math.ceil(arr.length / maxCount);
    const sampled: T[] = [];
    for (let i = 0; i < arr.length; i += step) sampled.push(arr[i]);
    return sampled;
}

const SankeyDiagram = React.forwardRef<HTMLDivElement, SankeyDiagramProps>(
    ({ data, zoomFactor = 1, hideTooltip = false }, ref) => {
        const svgRef = useRef<SVGSVGElement | null>(null);
        const containerRef = useRef<HTMLDivElement | null>(null);
        // No local data state, everything from props.data
        useImperativeHandle(ref, () => containerRef.current as HTMLDivElement);

        // Tooltip for Sankey sample hover
        let tooltip = d3.select("body").select("#global-tooltip");
        if (tooltip.empty()) {
            d3.select("body")
                .append("div")
                .attr("id", "global-tooltip")
                .style("position", "absolute")
                .style("padding", "4px")
                .style("background", "white")
                .style("border", "1px solid #ccc")
                .style("borderRadius", "4px")
                .style("pointerEvents", "none")
                .style("zIndex", "9999")
                .style("opacity", 0);
        }

        // Always reset style every mount (keeps tooltip consistent)
        tooltip
            .style("position", "absolute")
            .style("padding", "4px")
            .style("background", "white")
            .style("border", "1px solid #ccc")
            .style("borderRadius", "4px")
            .style("pointerEvents", "none")
            .style("zIndex", "9999")
            .style("opacity", 0)
            .html("");

        useEffect(() => {
            // helper to escape CSS class selectors for arbitrary ids
            const escapeForSelector = (s: string) => s.replace(/[^a-zA-Z0-9_-]/g, "\\$&");

            const applyHighlightToThisSankey = (ids: string[]) => {
                const svgSel = d3.select(svgRef.current);
                // dim main links and sample-trajectories
                svgSel.selectAll(".links path").attr("opacity", 0.1);
                svgSel.selectAll(".sample-trajectories line").attr("opacity", 0.05);
                // dim all sankey links globally on scatter highlight
                svgSel.selectAll(".sankey-links path")
                    .attr("opacity", 0.1);

                ids.forEach(id => {
                    const safeId = escapeForSelector(String(id));
                    svgSel.selectAll(`.sample-${safeId}`)
                        .attr("opacity", 1)
                        .attr("stroke-width", 1.8);
                });
            };

            const clearHighlight = () => {
                const svgSel = d3.select(svgRef.current);

                // Restore base sankey link opacity
                svgSel.selectAll(".sankey-links path")
                    .attr("opacity", 0.4);

                // Restore aggregated layer links
                svgSel.selectAll(".links path")
                    .attr("opacity", 0.4);

                // Hide all sample trajectories
                svgSel.selectAll(".sample-trajectories line")
                    .attr("opacity", 0)
                    .attr("stroke-width", 0.5);

                // Hide sample points
                svgSel.selectAll(".sample-points circle")
                    .attr("opacity", 0);

                // DO NOT force scatter to clear itself here — scatter handles its own clearing
                // Removed: window.dispatchEvent("scatter-highlight-clear")

                // reset global highlight state
                (window as any).__GLOBAL_SANKY_HIGHLIGHT_IDS = [];
            };

            const handleHighlight = (e: any) => {
                let ids: string[] = [];

                // Support both CustomEvent(detail = { ids, source }) and legacy detail = ids
                if (Array.isArray(e?.detail)) {
                    ids = e.detail;
                } else if (Array.isArray(e?.detail?.ids)) {
                    ids = e.detail.ids;
                }

                // store globally so newly mounted sankeys can pick it up
                (window as any).__GLOBAL_SANKY_HIGHLIGHT_IDS = ids;

                if (ids.length > 0) {
                    applyHighlightToThisSankey(ids);
                } else {
                    clearHighlight();
                }
            };

            window.addEventListener("sankey-highlight-ids", handleHighlight);
            window.addEventListener("sankey-highlight-clear", clearHighlight);

            return () => {
                window.removeEventListener("sankey-highlight-ids", handleHighlight);
                window.removeEventListener("sankey-highlight-clear", clearHighlight);
            };
        }, []);

        // Helper: Build graph from rows
        function buildGraphFromRows(
            rows: SankeyDiagramProps["data"]
        ): SankeyData {
            const nodeNames = new Set<string>();
            const linkMap = new Map<string, number>();
            if (!rows) throw Error("No rows (?)");

            rows.forEach(r => {
                const src = `${r.from_layer}_${r.from_cluster}`;
                const tgt = `${r.to_layer}_${r.to_cluster}`;
                nodeNames.add(src);
                nodeNames.add(tgt);

                const key = `${src}->${tgt}`;
                linkMap.set(key, (linkMap.get(key) ?? 0) + 1);
            });

            const nodes: Node[] = Array.from(nodeNames).map(name => ({ name }));
            const nodeIndex = new Map(nodes.map((n, i) => [n.name, i]));

            const links: Link[] = Array.from(linkMap.entries()).map(([key, value]) => {
                const [src, tgt] = key.split("->");
                return {
                    source: nodeIndex.get(src)!,
                    target: nodeIndex.get(tgt)!,
                    value,
                    key,
                } as any;
            });

            return { nodes, links };
        }

        // Rewritten drawSankey to use only data from props
        const drawSankey = (width: number, graph: SankeyData) => {
            let linkSamplesMap = new Map<string, string[]>();
            let nodeSamplesMap = new Map<string, string[]>();
            let currentHighlightedNode: string | null = null;

            // Build splitLabelMap from data
            const splitLabelMap = new Map<string, string>();
            if (!data) return;
            data.forEach(r => {
                splitLabelMap.set(String(r.real_id), r.split_label || "missing");
            });

            function highlightSamplesForNode(nodeName: string) {
                const samples = nodeSamplesMap.get(nodeName) || [];
                const svgSel = d3.select(svg.node());
                svgSel.selectAll(".links path")
                    .attr("stroke", "#ccc")
                    .attr("opacity", 0.2);
                svgSel.selectAll(".sample-trajectories line")
                    .attr("opacity", 0);
                svgSel.selectAll(".sample-points circle")
                    .attr("opacity", 0);
                samples.forEach(sample_id => {
                    svgSel.selectAll(`.sample-${sample_id}`)
                        .attr("opacity", 0.8);
                });
            }

            const svg = d3.select(svgRef.current);
            svg.selectAll("*").remove();

            const height = 400;
            const fittedWidth = Math.max(width, 200);

            // Margin calculation
            let layerNames: string[] | undefined;
            if (graph && graph.nodes && graph.nodes.length > 0) {
                const getLayerName = (fullName: string) => {
                    const idx = fullName.lastIndexOf("_");
                    return idx === -1 ? fullName : fullName.slice(0, idx);
                };
                const layerNameSet = new Set<string>();
                graph.nodes.forEach(n => {
                    if (n && n.name) {
                        layerNameSet.add(getLayerName(n.name));
                    }
                });
                layerNames = Array.from(layerNameSet).sort();
            }
            const estimatedLayers = layerNames?.length || 8;
            const margin = width / estimatedLayers / 2;

            const sankeyGen = sankey<Node, Link>()
                .nodeWidth(10)
                .nodePadding(7)
                .extent([
                    [margin, 1],
                    [width - margin, height - 5],
                ]);

            const { nodes: layoutNodes, links: layoutLinks } = sankeyGen({
                nodes: graph.nodes.map((d) => Object.assign({}, d)),
                links: graph.links.map((d) => Object.assign({}, d)),
            });

            svg.attr("width", width).attr("height", height);

            const getLayerName = (fullName: string) => {
                const idx = fullName.lastIndexOf("_");
                return idx === -1 ? fullName : fullName.slice(0, idx);
            };

            const layerNameSet = new Set<string>();
            layoutNodes.forEach(n => {
                const layerName = getLayerName(n.name);
                layerNameSet.add(layerName);
            });
            const layerNames2 = Array.from(layerNameSet).sort();

            // ---- External bottom labels aligned to scatterplot columns ----
            const externalLabelDiv = containerRef.current?.querySelector("#sankey-layer-labels");
            if (externalLabelDiv) {
                externalLabelDiv.innerHTML = "";
                const layerCount = layerNames2.length;
                const colWidth = width / layerCount;
                layerNames2.forEach((layerName, i) => {
                    const div = document.createElement("div");
                    div.style.flex = "0 0 " + colWidth + "px";
                    div.style.textAlign = "center";
                    div.textContent = layerName;
                    externalLabelDiv.appendChild(div);
                });
            }

            // Assign each layer a color
            const layerColorMap = new Map<string, string>();
            layerNames2.forEach((layerName, i) => {
                layerColorMap.set(
                    layerName,
                    d3.schemeTableau10[i % d3.schemeTableau10.length]
                );
            });

            // Sort nodes within each layer by y0
            const nodesByLayer = new Map<string, Node[]>();
            layerNames2.forEach(layerName => {
                const nodesInLayer = layoutNodes.filter(n => getLayerName(n.name) === layerName);
                nodesInLayer.sort((a, b) => (a.y0 ?? 0) - (b.y0 ?? 0));
                nodesByLayer.set(layerName, nodesInLayer);
            });

            // Draw aggregated links as colored wide paths, vertically separated by split
            const linkGroup = svg
                .append("g")
                .attr("fill", "none")
                .attr("class", "sankey-links");

            // ----------- Split-Link Generation using link.y0/y1 and link.width -----------
            const splits = ["train", "test", "experimental", "missing"] as const;
            const splitColors: Record<(typeof splits)[number], string> = {
                train: "#1f77b4",
                test: "#ff7f0e",
                experimental: "#2ca02c",
                missing: "#999999",
            };

            // Build linkSamplesMap from data
            linkSamplesMap.clear();
            layoutLinks.forEach((link: any) => {
                const srcName = (link.source as Node).name;
                const tgtName = (link.target as Node).name;
                const key = `${srcName}->${tgtName}`;
                linkSamplesMap.set(key, []);
            });
            // Fill linkSamplesMap from data
            data.forEach(r => {
                const sid = String(r.real_id);
                const srcName = `${r.from_layer}_${r.from_cluster}`;
                const tgtName = `${r.to_layer}_${r.to_cluster}`;
                const key = `${srcName}->${tgtName}`;
                if (linkSamplesMap.has(key)) {
                    linkSamplesMap.get(key)!.push(sid);
                }
            });

            const separatedLinks: {
                sx: number;
                sy0: number;
                sy1: number;
                tx: number;
                ty0: number;
                ty1: number;
                width: number;
                stroke: string;
            }[] = [];

            layoutLinks.forEach((link: any) => {
                const srcNode = link.source as Node;
                const tgtNode = link.target as Node;
                const key = `${srcNode.name}->${tgtNode.name}`;
                const sampleIds = linkSamplesMap.get(key) || [];
                // count train / test / experimental / missing for this link

                const counts: Record<Split, number> = {
                    train: 0,
                    test: 0,
                    experimental: 0,
                    missing: 0,
                };

                sampleIds.forEach((id) => {
                    // cast the map result to Split and fallback to "missing"
                    const label = (splitLabelMap.get(String(id)) as Split) || "missing";
                    // now safe to index counts with label
                    counts[label] += 1;
                });
                const totalCount =
                    counts.train + counts.test + counts.experimental + counts.missing;
                if (totalCount === 0) return;
                const linkCenterY0 = Number(link.y0);
                const linkCenterY1 = Number(link.y1);
                const linkWidth = Number(link.width ?? totalCount);
                if (!isFinite(linkCenterY0) || !isFinite(linkCenterY1) || !isFinite(linkWidth)) {
                    return;
                }
                const sx = Number(srcNode.x1);
                const tx = Number(tgtNode.x0);
                const baseOffset = -linkWidth / 2;
                let acc = 0;
                splits.forEach((sp) => {
                    const cnt = counts[sp];
                    if (cnt <= 0) return;
                    const segWidth = (linkWidth * cnt) / totalCount;
                    const centerOffset = baseOffset + acc + segWidth / 2;
                    const sy = linkCenterY0 + centerOffset;
                    const ty = linkCenterY1 + centerOffset;
                    separatedLinks.push({
                        sx,
                        sy0: sy,
                        sy1: sy,
                        tx,
                        ty0: ty,
                        ty1: ty,
                        width: segWidth,
                        stroke: splitColors[sp],
                    });
                    acc += segWidth;
                });
            });

            // Draw the split links as cubic Bézier curves per segment
            linkGroup
                .selectAll("path")
                .data(separatedLinks)
                .join("path")
                .attr("d", (d) => {
                    const sx = d.sx;
                    const sy = d.sy0;
                    const tx = d.tx;
                    const ty = d.ty0;
                    const mx = (sx + tx) / 2;
                    return `M${sx},${sy} C${mx},${sy} ${mx},${ty} ${tx},${ty}`;
                })
                .attr("stroke", (d) => d.stroke)
                .attr("stroke-width", (d) => d.width)
                .attr("fill", "none")
                .attr("opacity", 0.4);

            // ----------- New: Draw sample-level points and trajectories -----------
            // Build samplePathsMap: real_id -> array of node names (path)
            const samplePathsMap = new Map<string, string[]>();
            data.forEach((r) => {
                const sid = String(r.real_id);
                const fromNode = `${r.from_layer}_${r.from_cluster}`;
                const toNode = `${r.to_layer}_${r.to_cluster}`;
                if (!samplePathsMap.has(sid)) samplePathsMap.set(sid, []);
                const path = samplePathsMap.get(sid)!;
                if (path[path.length - 1] !== fromNode) path.push(fromNode);
                path.push(toNode);
            });
            // For each node, collect samples assigned to it
            nodeSamplesMap = new Map<string, string[]>();
            samplePathsMap.forEach((nodeNames, real_id) => {
                nodeNames.forEach(nodeName => {
                    if (!nodeSamplesMap.has(nodeName)) {
                        nodeSamplesMap.set(nodeName, []);
                    }
                    nodeSamplesMap.get(nodeName)!.push(real_id);
                });
            });
            // Build map of nodeName -> incoming links (source node names)
            const nodeIncomingMap = new Map<string, string[]>();
            layoutLinks.forEach(link => {
                const targetNode = (link.target as Node).name;
                const sourceNode = (link.source as Node).name;
                if (!nodeIncomingMap.has(targetNode)) {
                    nodeIncomingMap.set(targetNode, []);
                }
                nodeIncomingMap.get(targetNode)!.push(sourceNode);
            });
            // Build map of nodeName -> outgoing links (target node names)
            const nodeOutgoingMap = new Map<string, string[]>();
            layoutLinks.forEach(link => {
                const sourceNode = (link.source as Node).name;
                const targetNode = (link.target as Node).name;
                if (!nodeOutgoingMap.has(sourceNode)) {
                    nodeOutgoingMap.set(sourceNode, []);
                }
                nodeOutgoingMap.get(sourceNode)!.push(targetNode);
            });
            // For each node, assign y positions for sample points inside node vertical range [y0, y1]
            // Map real_id -> Map nodeName -> y position
            const sampleNodeYMap = new Map<string, Map<string, number>>();
            // Build map nodeName -> node layout object for quick lookup
            const nodeLayoutMap = new Map<string, Node>();
            layoutNodes.forEach(n => nodeLayoutMap.set(n.name, n));
            function getDownstreamY(real_id: string, nodeName: string): number[] {
                const outgoingNodes = nodeOutgoingMap.get(nodeName);
                if (!outgoingNodes) return [];
                const ys: number[] = [];
                outgoingNodes.forEach(outNodeName => {
                    const y = sampleNodeYMap.get(real_id)?.get(outNodeName);
                    if (y !== undefined) ys.push(y);
                });
                return ys;
            }
            // Infer pathLayerNames as the sequence of unique layer names in the first path
            let pathLayerNames: string[] = [];
            if (samplePathsMap.size > 0) {
                const firstPath = Array.from(samplePathsMap.values())[0];
                pathLayerNames = firstPath.map(nm => {
                    const idx = nm.lastIndexOf("_");
                    return idx === -1 ? nm : nm.slice(0, idx);
                });
            }
            for (let layerIndex = pathLayerNames.length - 1; layerIndex >= 0; layerIndex--) {
                const layerName = pathLayerNames[layerIndex];
                const nodesInLayer = layoutNodes.filter(n => getLayerName(n.name) === layerName);
                nodesInLayer.forEach(node => {
                    const samples = nodeSamplesMap.get(node.name) || [];
                    if (samples.length === 0) return;
                    const sampleDownstreamYMap = new Map<string, number | null>();
                    samples.forEach(real_id => {
                        const downstreamYs = getDownstreamY(real_id, node.name);
                        if (downstreamYs.length > 0) {
                            const avgY = d3.mean(downstreamYs) ?? null;
                            sampleDownstreamYMap.set(real_id, avgY);
                        } else {
                            sampleDownstreamYMap.set(real_id, null);
                        }
                    });
                    samples.sort((a, b) => {
                        const aDown = sampleDownstreamYMap.get(a);
                        const bDown = sampleDownstreamYMap.get(b);
                        if (aDown !== null && bDown !== null) {
                            return (aDown ?? 0) - (bDown ?? 0);
                        } else if (aDown !== null) {
                            return -1;
                        } else if (bDown !== null) {
                            return 1;
                        } else {
                            const numA = Number(a), numB = Number(b);
                            if (!isNaN(numA) && !isNaN(numB)) return numA - numB;
                            return a < b ? -1 : a > b ? 1 : 0;
                        }
                    });
                    const count = samples.length;
                    const y0 = node.y0 ?? 0;
                    const y1 = node.y1 ?? 0;
                    const nodeHeight = y1 - y0;
                    samples.forEach((real_id, i) => {
                        const y = count === 1 ? y0 + nodeHeight / 2 : y0 + (i + 0.5) * (nodeHeight / count);
                        if (!sampleNodeYMap.has(real_id)) {
                            sampleNodeYMap.set(real_id, new Map());
                        }
                        sampleNodeYMap.get(real_id)!.set(node.name, y);
                    });
                });
            }
            // Prepare group for sample trajectories (per-link lines)
            const samplesGroup = svg.append("g").attr("class", "sample-trajectories");
            // Build a map from link key to sample IDs that traverse that link
            samplePathsMap.forEach((nodeNames, real_id) => {
                for (let i = 0; i < nodeNames.length - 1; i++) {
                    const sourceNodeName = nodeNames[i];
                    const targetNodeName = nodeNames[i + 1];
                    const linkKey = `${sourceNodeName}->${targetNodeName}`;
                    if (!linkSamplesMap.has(linkKey)) {
                        linkSamplesMap.set(linkKey, []);
                    }
                    linkSamplesMap.get(linkKey)!.push(real_id);
                }
            });
            // Batch build all sample line data for D3 join
            const lineData: {
                x1: number, y1: number, x2: number, y2: number,
                stroke: string, className: string
            }[] = [];
            layoutLinks.forEach(link => {
                const sourceNode = link.source as Node;
                const targetNode = link.target as Node;
                const linkKey = `${sourceNode.name}->${targetNode.name}`;
                const samplesOnLink = strideSample(
                    linkSamplesMap.get(linkKey) || [],
                    MAX_TRAJECTORIES_PER_LINK
                );
                if (samplesOnLink.length === 0) return;
                samplesOnLink.sort((a, b) => {
                    const yA = sampleNodeYMap.get(a)?.get(sourceNode.name) ?? 0;
                    const yB = sampleNodeYMap.get(b)?.get(sourceNode.name) ?? 0;
                    return yA - yB;
                });
                samplesOnLink.forEach(real_id => {
                    const ySource = sampleNodeYMap.get(real_id)?.get(sourceNode.name);
                    const yTarget = sampleNodeYMap.get(real_id)?.get(targetNode.name);
                    if (ySource === undefined || yTarget === undefined) return;
                    const xSource = (sourceNode.x0 ?? 0) + sankeyGen.nodeWidth() / 2;
                    const xTarget = (targetNode.x0 ?? 0) + sankeyGen.nodeWidth() / 2;
                    const safeLinkClass = `link-${linkKey.replace(/->/g, "-").replace(/_/g, "_")}`;
                    const sampleClass = `sample-${real_id}`;
                    const rawLabel = splitLabelMap.get(real_id) || "missing";
                    let c = "#999999";
                    if (rawLabel === "train") c = "#1f77b4";
                    else if (rawLabel === "test") c = "#ff7f0e";
                    else if (rawLabel === "experimental") c = "#2ca02c";
                    let strokeColor = c;
                    lineData.push({
                        x1: xSource, y1: ySource, x2: xTarget, y2: yTarget,
                        stroke: strokeColor,
                        className: `${safeLinkClass} ${sampleClass}`
                    });
                });
            });
            const limitedLineData = strideSample(lineData, MAX_TRAJECTORIES_TOTAL);
            samplesGroup.selectAll<SVGLineElement, LineDatum>("line")
                .data(limitedLineData)
                .join("line")
                .attr("x1", d => d.x1)
                .attr("y1", d => d.y1)
                .attr("x2", d => d.x2)
                .attr("y2", d => d.y2)
                .attr("stroke", d => d.stroke)
                .attr("stroke-width", 0.5)
                .attr("opacity", 0)
                .attr("class", d => d.className)
                .attr("data-linkkey", d => d.className.split(" ")[0])
                .on("mouseover", function(this: SVGLineElement, event: any, d: LineDatum) {
                    const classes = d.className.split(" ");
                    const sampleClass = classes.find(c => c.startsWith("sample-"));
                    if (!sampleClass) return;
                    const id = sampleClass.replace("sample-", "");
                    if (!hideTooltip) {
                        tooltip
                            .html(`<img src="/data/molecule_images/${id}_ligand.png" style="width:200px;height:auto;" />`)
                            .style("left", (event.pageX + 12) + "px")
                            .style("top", (event.pageY + 12) + "px")
                            .style("opacity", 1);
                    }
                })
                .on("mouseout", function() {
                    tooltip.style("opacity", 0);
                });
            // ----------- Multi‑color stacked node rectangles (train/test/experimental/missing) -----------
            const nodeGroup = svg.append("g").attr("class", "node-stacks");
            nodeGroup
                .selectAll<SVGGElement, Node>("g.node")
                .data(layoutNodes as Node[])
                .join("g")
                .attr("class", "node")
                .each(function(this: SVGGElement, nodeData: Node) {
                    const g = d3.select(this as SVGGElement);
                    // Collect split counts for this node from samples
                    const sampleIds = nodeSamplesMap.get(nodeData.name) || [];
                    const counts: Record<Split, number> = {
                        train: 0,
                        test: 0,
                        experimental: 0,
                        missing: 0,
                    };

                    sampleIds.forEach((id) => {
                        // cast the map result to Split and fallback to "missing"
                        const label = (splitLabelMap.get(String(id)) as Split) || "missing";
                        // now safe to index counts with label
                        counts[label] += 1;
                    });

                    const total = counts.train + counts.test + counts.experimental + counts.missing;
                    const nodeHeight = (nodeData.y1 ?? 0) - (nodeData.y0 ?? 0);
                    const nodeX = (nodeData.x0 ?? 0);
                    const nodeY = (nodeData.y0 ?? 0);

                    const nodeWidth = sankeyGen.nodeWidth();
                    if (total === 0) {
                        g.append("rect")
                            .attr("x", nodeX)
                            .attr("y", nodeY)
                            .attr("width", nodeWidth)
                            .attr("height", nodeHeight)
                            .attr("fill", "#ccc");
                        return;
                    }
                    const splits: Split[] = ["train", "test", "experimental", "missing"];
                    const colors = {
                        train: "#1f77b4",
                        test: "#ff7f0e",
                        experimental: "#2ca02c",
                        missing: "#999999",
                    };
                    let accY = nodeData.y0 ?? 0;
                    splits.forEach(sp => {
                        const c = counts[sp];
                        if (c <= 0) return;
                        const h = (c / total) * nodeHeight;
                        g.append("rect")
                            .attr("x", nodeX)
                            .attr("y", accY)
                            .attr("width", nodeWidth)
                            .attr("height", h)
                            .attr("fill", colors[sp]);
                        accY += h;
                    });

                    g.style("cursor", "pointer")
                        .on("mouseover", function() {
                            d3.select(this).selectAll("rect")
                                .attr("stroke", "black")
                                .attr("stroke-width", 2);
                        })
                        .on("mouseout", function() {
                            d3.select(this).selectAll("rect")
                                .attr("stroke", "none");
                        })
                        .on("click", function(this: SVGGElement, event: any) {
                            const d = (d3.select(this as SVGGElement).datum() as Node);
                            const samples = nodeSamplesMap.get(d.name) || [];
                            if (currentHighlightedNode === d.name) {
                                const svgSel = d3.select(svg.node());
                                svgSel.selectAll(".sankey-links path")
                                    .attr("stroke", (d: any) => d.stroke)
                                    .attr("opacity", 0.4);
                                svgSel.selectAll(".sample-trajectories line").attr("opacity", 0);
                                svgSel.selectAll(".sample-points circle").attr("opacity", 0);
                                currentHighlightedNode = null;
                                window.dispatchEvent(new CustomEvent("sankey-highlight-clear"));
                                window.dispatchEvent(new CustomEvent("scatter-highlight-clear"));
                                return;
                            }
                            currentHighlightedNode = d.name;
                            const svgSel = d3.select(svg.node());
                            svgSel.selectAll(".sankey-links path").attr("opacity", 0.1);
                            highlightSamplesForNode(d.name);
                            window.dispatchEvent(
                                new CustomEvent("scatter-highlight-ids", { detail: samples })
                            );
                            window.dispatchEvent(
                                new CustomEvent("sankey-highlight-ids", { detail: samples })
                            );
                        });
                });
            // Draw node labels
            svg
                .append("g")
                .selectAll("text")
                .data(layoutNodes)
                .join("text")
                .attr("x", (d) => (d.x0 ?? 0) - 6)
                .attr("y", (d) => ((d.y1 ?? 0) + (d.y0 ?? 0)) / 2)
                .attr("dy", "0.35em");
            // Draw sample points as small circles (disabled)
            // const samplePointsGroup = svg.append("g").attr("class", "sample-points");
            // sampleNodeYMap.forEach((nodeYMap, real_id) => {
            //   nodeYMap.forEach((y, nodeName) => {
            //     const node = nodeLayoutMap.get(nodeName);
            //     if (!node) return;
            //     const x = (node.x0 ?? 0) + sankeyGen.nodeWidth() / 2;
            //     const layerName = getLayerName(node.name);
            //     const color = layerColorMap.get(layerName) ?? "#999";
            //     // samplePointsGroup.append("circle")
            //     //   .attr("cx", x)
            //     //   .attr("cy", y)
            //     //   .attr("r", 2.5)
            //     //   .attr("fill", color)
            //     //   .attr("stroke", "none");
            //   });
            // });
            // If there is an active global highlight (from other windows), apply it now
            const globalIds: string[] = (window as any).__GLOBAL_SANKY_HIGHLIGHT_IDS || [];
            if (globalIds.length > 0) {
                const escapeForSelector = (s: string) => s.replace(/[^a-zA-Z0-9_-]/g, "\\$&");
                svg.selectAll(".links path").attr("opacity", 0.1);
                svg.selectAll(".sample-trajectories line").attr("opacity", 0.05);
                globalIds.forEach(id => {
                    const safeId = escapeForSelector(String(id));
                    svg.selectAll(`.sample-${safeId}`).attr("opacity", 1).attr("stroke-width", 1.8);
                });
            }
        };

        useEffect(() => {
            if (!data || data.length === 0) return;
            const container = containerRef.current;
            if (!container) return;

            const graph = buildGraphFromRows(data);
            let rafId: number | null = null;

            const draw = () => {
                drawSankey(container.clientWidth, graph);
            };
            const scheduleDraw = () => {
                if (rafId !== null) cancelAnimationFrame(rafId);
                rafId = requestAnimationFrame(() => {
                    draw();
                    rafId = null;
                });
            };

            draw();
            const resizeObserver = new ResizeObserver(scheduleDraw);
            resizeObserver.observe(container);
            return () => {
                resizeObserver.disconnect();
                if (rafId !== null) cancelAnimationFrame(rafId);
            };
        }, [data]);

        return (
            <div
                ref={containerRef}
                style={{ width: "100%", height: "430px", position: "relative", overflowX: "scroll" }}
                onScroll={(e) => {
                    const left = (e.target as HTMLDivElement).scrollLeft;
                    window.dispatchEvent(new CustomEvent("sankey-scroll", { detail: left }));
                }}
                onWheel={(e) => {
                    // Horizontal wheel sync with ScatterPlot
                    const container = e.currentTarget as HTMLDivElement;
                    container.scrollLeft += e.deltaY;
                    window.dispatchEvent(new CustomEvent("sankey-scroll", { detail: container.scrollLeft }));
                }}
            >
                <svg
                    ref={svgRef}
                    style={{
                        width: `${zoomFactor * 100}%`,
                        minWidth: `${zoomFactor * 100}%`
                    }}
                ></svg>

                <div
                    id="sankey-layer-labels"
                    style={{
                        width: "100%",
                        display: "flex",
                        justifyContent: "space-between",
                        textAlign: "center",
                        marginTop: "4px",
                        fontSize: "12px"
                    }}
                ></div>
            </div>
        );
    }
);
export default SankeyDiagram;
