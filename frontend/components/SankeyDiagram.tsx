"use client";
import * as d3 from "d3";
import {
  sankey,
  sankeyLinkHorizontal,
  SankeyNodeMinimal,
  SankeyLinkMinimal,
} from "d3-sankey";
import React, { useEffect, useRef, useState } from "react";

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
  sampleFile: string;
}

export default function SankeyDiagram({ clusterFile, sampleFile }: SankeyDiagramProps) {
  const ref = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [data, setData] = useState<SankeyData | null>(null);
  // Store mapping from real_id to split label ("train", "test", "experimental", etc)
  const [splitLabelMap, setSplitLabelMap] = useState<Map<string, string>>(new Map());

  // Tooltip for Sankey sample hover
  const tooltip = d3.select("body")
    .append("div")
    .attr("class", "sankey-tooltip")
    .style("position", "absolute")
    .style("padding", "4px")
    .style("background", "white")
    .style("border", "1px solid #ccc")
    .style("borderRadius", "4px")
    .style("pointerEvents", "none")
    .style("zIndex", 9999)
    .style("opacity", 0);

  useEffect(() => {
    // helper to escape CSS class selectors for arbitrary ids
    const escapeForSelector = (s: string) => s.replace(/[^a-zA-Z0-9_-]/g, "\\$&");

    const applyHighlightToThisSankey = (ids: string[]) => {
      const svgSel = d3.select(ref.current);
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
      const svgSel = d3.select(ref.current);

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
      const ids: string[] = e?.detail || [];
      // store globally so newly mounted sankeys can pick it up
      (window as any).__GLOBAL_SANKY_HIGHLIGHT_IDS = ids;
      applyHighlightToThisSankey(ids);
    };

    window.addEventListener("sankey-highlight-ids", handleHighlight);
    window.addEventListener("sankey-highlight-clear", clearHighlight);

    return () => {
      window.removeEventListener("sankey-highlight-ids", handleHighlight);
      window.removeEventListener("sankey-highlight-clear", clearHighlight);
    };
  }, []);

  useEffect(() => {
    d3.csv(sampleFile).then((rows: any[]) => {
      // rows: real_id,from_layer,to_layer,from_cluster,to_cluster
      const links: Link[] = [];
      const nodeNames = new Set<string>();

      const groups = d3.group(rows, r => r.real_id);

      groups.forEach(records => {
        records.forEach((cur: any) => {
          const fromNode = `${cur.from_layer}_${cur.from_cluster}`;
          const toNode = `${cur.to_layer}_${cur.to_cluster}`;

          nodeNames.add(fromNode);
          nodeNames.add(toNode);

          const key = `${fromNode}->${toNode}`;
          const existing = links.find((l: any) => l.key === key);
          if (existing) existing.value += 1;
          else links.push({ source: fromNode, target: toNode, value: 1, key });
        });
      });

      const nodes = Array.from(nodeNames).map(name => ({ name }));
      const nameToIndex = new Map(nodes.map((n, i) => [n.name, i]));
      links.forEach((l: any) => {
        l.source = nameToIndex.get(l.source);
        l.target = nameToIndex.get(l.target);
      });
      setData({ nodes, links });
    });
  }, [sampleFile]);

  useEffect(() => {
    d3.csv("/data/cluster_transition_matrices_1_individual_real_id.csv")
      .then((rows2: any[]) => {
        const map = new Map<string, string>();
        rows2.forEach(r => {
          map.set(String(r.real_id), r.split_label || "missing");
        });
        setSplitLabelMap(map);
      });
  }, []);

  const drawSankey = (width: number, data: SankeyData) => {
    let linkSamplesMap = new Map<string, string[]>();
    // highlight samples for a given node
    // Global nodeSamplesMap for sharing across highlightSamplesForNode and CSV parsing
    let nodeSamplesMap = new Map<string, string[]>();
    let currentHighlightedNode: string | null = null;

    function highlightSamplesForNode(nodeName: string) {
      const samples = nodeSamplesMap.get(nodeName) || [];
      const svgSel = d3.select(svg.node());
      svgSel.selectAll(".links path")
        // .transition().duration(200)
        .attr("stroke", "#ccc")
        .attr("opacity", 0.2);

      svgSel.selectAll(".sample-trajectories line")
        // .transition().duration(200)
        .attr("opacity", 0);
      svgSel.selectAll(".sample-points circle")
        // .transition().duration(200)
        .attr("opacity", 0);

      samples.forEach(sample_id => {
        svgSel.selectAll(`.sample-${sample_id}`)
          // .transition().duration(300)
          .attr("opacity", 0.8);
      });
    }

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const height = 600;
    // Always fit SVG inside window with a min width, no overflow
    const fittedWidth = Math.max(width, 200);

    const sankeyGen = sankey<Node, Link>()
      .nodeWidth(10)
      .nodePadding(7)
      .extent([
        [1, 1],
        [width - 20, height - 5], // compress to fit inside available width
      ]);

    const { nodes: layoutNodes, links: layoutLinks } = sankeyGen({
      nodes: data.nodes.map((d) => Object.assign({}, d)),
      links: data.links.map((d) => Object.assign({}, d)),
    });

    svg.attr("width", fittedWidth).attr("height", height);

    // helper to extract layer name from node name
    const getLayerName = (fullName: string) => {
      const idx = fullName.lastIndexOf("_");
      return idx === -1 ? fullName : fullName.slice(0, idx);
    };

    // Derive layer names from node names
    const layerNameSet = new Set<string>();
    layoutNodes.forEach(n => {
      const layerName = getLayerName(n.name);
      layerNameSet.add(layerName);
    });
    const layerNames = Array.from(layerNameSet).sort();

    // Assign each layer a distinct hue around the color wheel using HSL
    const layerColorMap = new Map<string, string>();
    layerNames.forEach((layerName, i) => {
      layerColorMap.set(
        layerName,
        d3.schemeTableau10[i % d3.schemeTableau10.length] // ✅ 不加括号，直接取数组元素
      );
    });

    // Sort nodes within each layer by y0 (top to bottom)
    const nodesByLayer = new Map<string, Node[]>();
    layerNames.forEach(layerName => {
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

    // Build linkSamplesMap from sampleFile once, then compute per-link split counts
    linkSamplesMap.clear();
    layoutLinks.forEach((link: any) => {
      const srcName = (link.source as Node).name;
      const tgtName = (link.target as Node).name;
      const key = `${srcName}->${tgtName}`;
      linkSamplesMap.set(key, []);
    });

    d3.csv(sampleFile).then((rows: any[]) => {
      rows.forEach((r: any) => {
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
        const counts: Record<(typeof splits)[number], number> = {
          train: 0,
          test: 0,
          experimental: 0,
          missing: 0,
        };
        sampleIds.forEach((id) => {
          const label = (splitLabelMap.get(String(id)) as (typeof splits)[number]) || "missing";
          if (counts[label] !== undefined) counts[label] += 1;
          else counts.missing += 1;
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
    });



    // ----------- New: Draw sample-level points and trajectories -----------

    d3.csv(sampleFile).then(rows => {
      linkSamplesMap.clear();
      // Use new real_id, from_layer, to_layer, from_cluster, to_cluster fields
      const samplePathsMap = new Map<string, string[]>();

      rows = rows as any[];
      rows.forEach((r: any) => {
        const sid = r.real_id;
        const fromNode = `${r.from_layer}_${r.from_cluster}`;
        const toNode = `${r.to_layer}_${r.to_cluster}`;

        if (!samplePathsMap.has(sid)) samplePathsMap.set(sid, []);
        const path = samplePathsMap.get(sid)!;
        // avoid duplicate consecutive nodes
        if (path[path.length - 1] !== fromNode) path.push(fromNode);
        path.push(toNode);
      });

      // For each node, collect samples assigned to it
      // Map nodeName -> array of real_ids
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
      // But first sort samples by upstream average y positions if upstream info exists,
      // otherwise fallback to sorting by sample_id numerically

      // Map real_id -> Map nodeName -> y position (for previous layer)
      // We will assign sampleNodeYMap progressively from first layer to last layer
      const sampleNodeYMap = new Map<string, Map<string, number>>();

      // Build map nodeName -> node layout object for quick lookup
      const nodeLayoutMap = new Map<string, Node>();
      layoutNodes.forEach(n => nodeLayoutMap.set(n.name, n));

      // Helper to get downstream sample y positions for a sample at a node
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

      // Process layers in reverse order (output-oriented, downstream-based)
      // We need to reconstruct pathLayerNames from the samplePathsMap
      // (since it's now an array of node names per real_id)
      // We'll infer pathLayerNames as the sequence of unique layer names in the paths
      let pathLayerNames: string[] = [];
      if (samplePathsMap.size > 0) {
        // Get the first path
        const firstPath = Array.from(samplePathsMap.values())[0];
        pathLayerNames = firstPath.map(nm => {
          const idx = nm.lastIndexOf("_");
          return idx === -1 ? nm : nm.slice(0, idx);
        });
      }

      for (let layerIndex = pathLayerNames.length - 1; layerIndex >= 0; layerIndex--) {
        const layerName = pathLayerNames[layerIndex];
        // Get nodes in this layer
        const nodesInLayer = layoutNodes.filter(n => getLayerName(n.name) === layerName);

        nodesInLayer.forEach(node => {
          const samples = nodeSamplesMap.get(node.name) || [];
          if (samples.length === 0) return;

          // For each sample, compute average downstream y position if any
          // Map real_id -> avg downstream y
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

          // Sort samples: those with downstream info by avg downstream y ascending,
          // those without downstream info by numeric real_id ascending
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
              // fallback to numeric real_id sort
              const numA = Number(a), numB = Number(b);
              if (!isNaN(numA) && !isNaN(numB)) return numA - numB;
              return a < b ? -1 : a > b ? 1 : 0;
            }
          });

          // Assign y positions evenly spaced vertically within node height
          const count = samples.length;
          const y0 = node.y0 ?? 0;
          const y1 = node.y1 ?? 0;
          const height = y1 - y0;
          samples.forEach((real_id, i) => {
            const y = count === 1 ? y0 + height / 2 : y0 + (i + 0.5) * (height / count);
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
      // link key format: sourceNodeName->targetNodeName (matches layoutLinks key)

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
        const linkKey = link.key ?? `${sourceNode.name}->${targetNode.name}`;
        const samplesOnLink = linkSamplesMap.get(linkKey) || [];
        if (samplesOnLink.length === 0) return;

        // Sort samples by source node y position ascending
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


          console.log(typeof([...splitLabelMap.keys()][0]))
          // Compose CSS class: link key plus sample IDs
          // E.g., class="link-A_B sample-3"
          // To keep class names simple, replace '->' with '-' and remove underscores
          const safeLinkClass = `link-${linkKey.replace(/->/g, "-").replace(/_/g, "_")}`;
          const sampleClass = `sample-${real_id}`;
          // Determine stroke color by splitLabelMap
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

      samplesGroup.selectAll("line")
        .data(lineData)
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
        .on("mouseover", function(event, d) {
            const classes = d.className.split(" ");
            const sampleClass = classes.find(c => c.startsWith("sample-"));
            if (!sampleClass) return;
            const id = sampleClass.replace("sample-", "");

            tooltip
              .html(`<img src="/data/molecule_images/${id}_ligand.png" style="width:200px;height:auto;" />`)
              .style("left", (event.pageX + 12) + "px")
              .style("top", (event.pageY + 12) + "px")
              // .transition()
              // .duration(120)
              .style("opacity", 1);
        })
        .on("mouseout", function() {
            tooltip.style("opacity", 0);
        });
      // ----------- Multi‑color stacked node rectangles (train/test/experimental/missing) -----------
      const nodeGroup = svg.append("g").attr("class", "node-stacks");

      nodeGroup
        .selectAll("g.node")
        .data(layoutNodes)
        .join("g")
        .attr("class", "node")
        .each(function (nodeData) {

          const g = d3.select(this);

          // Collect split counts for this node from samples
          const sampleIds = nodeSamplesMap.get(nodeData.name) || [];

          const counts = { train: 0, test: 0, experimental: 0, missing: 0 };
          sampleIds.forEach(id => {
            const label = splitLabelMap.get(String(id)) || "missing";
            if (counts[label] !== undefined) counts[label] += 1;
            else counts["missing"] += 1;
          });

          const total = counts.train + counts.test + counts.experimental + counts.missing;
          const nodeHeight = (nodeData.y1 ?? 0) - (nodeData.y0 ?? 0);
          const nodeX = (nodeData.x0 ?? 0);
          const nodeWidth = sankeyGen.nodeWidth();

          if (total === 0) {
            // fallback single gray block when no samples
            g.append("rect")
              .attr("x", nodeX)
              .attr("y", nodeData.y0)
              .attr("width", nodeWidth)
              .attr("height", nodeHeight)
              .attr("fill", "#ccc");
            return;
          }

          const splits = ["train", "test", "experimental", "missing"];
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

          // pointer / click behaviors preserved
          g.style("cursor", "pointer")
            .on("mouseover", function () {
              d3.select(this).selectAll("rect")
                .attr("stroke", "black")
                .attr("stroke-width", 2);
            })
            .on("mouseout", function () {
              d3.select(this).selectAll("rect")
                .attr("stroke", "none");
            })
            .on("click", (event, d) => {

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
        .attr("dy", "0.35em")
        // .attr("text-anchor", "end")
        // .text((d) => d.name)
        // .filter((d) => (d.x0 ?? 0) < width / 2)
        // .attr("x", (d) => (d.x1 ?? 0) + 6)
        // .attr("text-anchor", "start");

      // Draw sample points as small circles
      const samplePointsGroup = svg.append("g").attr("class", "sample-points");

      sampleNodeYMap.forEach((nodeYMap, real_id) => {
        nodeYMap.forEach((y, nodeName) => {
          const node = nodeLayoutMap.get(nodeName);
          if (!node) return;
          const x = (node.x0 ?? 0) + sankeyGen.nodeWidth() / 2;
          const layerName = getLayerName(node.name);
          const color = layerColorMap.get(layerName) ?? "#999";

          // samplePointsGroup.append("circle")
          //   .attr("cx", x)
          //   .attr("cy", y)
          //   .attr("r", 2.5)
          //   .attr("fill", color)
          //   .attr("stroke", "none");
        });
      });
    });

    // If there is an active global highlight (from other windows), apply it now so
    // newly created sankey instances reflect the selection.
    const globalIds: string[] = (window as any).__GLOBAL_SANKY_HIGHLIGHT_IDS || [];
    if (globalIds.length > 0) {
      // use the same helper as the event handler: escape selectors
      const escapeForSelector = (s: string) => s.replace(/[^a-zA-Z0-9_-]/g, "\\$&");
      // dim base elements
      svg.selectAll(".links path").attr("opacity", 0.1);
      svg.selectAll(".sample-trajectories line").attr("opacity", 0.05);
      globalIds.forEach(id => {
        const safeId = escapeForSelector(String(id));
        svg.selectAll(`.sample-${safeId}`).attr("opacity", 1).attr("stroke-width", 1.8);
      });
    }
  };

  useEffect(() => {
    if (!data) return;
    const container = containerRef.current;
    if (!container) return;

    const draw = () => drawSankey(container.clientWidth, data);
    draw();

    const resizeObserver = new ResizeObserver(draw);
    resizeObserver.observe(container);
    return () => resizeObserver.disconnect();
  }, [data]);

  return (
    <div ref={containerRef} style={{ width: "100%", height: "600px", position: "relative", overflowX: "hidden" }}>
      <svg ref={ref}  style={{ width: "100%"}}></svg>
    </div>
  );
}