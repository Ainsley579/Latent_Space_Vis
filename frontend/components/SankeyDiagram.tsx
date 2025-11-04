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
  clusterFile: string;
  sampleFile: string;
}

export default function SankeyDiagram({ clusterFile, sampleFile }: SankeyDiagramProps) {
  const ref = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [data, setData] = useState<SankeyData | null>(null);

  useEffect(() => {
    d3.csv(sampleFile).then((rows) => {
      const layerNames = Object.keys(rows[0]).filter(k => k !== "sample_id");
      const links: Link[] = [];
      const nodeNames = new Set<string>();

      // Build links and nodes from sample_component_paths.csv
      for (let i = 0; i < rows.length; i++) {
        for (let j = 0; j < layerNames.length - 1; j++) {
          const fromLayer = layerNames[j];
          const toLayer = layerNames[j + 1];
          const fromNode = `${fromLayer}_${rows[i][fromLayer]}`;
          const toNode = `${toLayer}_${rows[i][toLayer]}`;
          nodeNames.add(fromNode);
          nodeNames.add(toNode);

          const key = `${fromNode}->${toNode}`;
          const existing = links.find((l: any) => l.key === key);
          if (existing) existing.value += 1;
          else links.push({ source: fromNode, target: toNode, value: 1, key });
        }
      }

      const nodes = Array.from(nodeNames).map((name) => ({ name }));
      const nameToIndex = new Map(nodes.map((n, i) => [n.name, i]));
      links.forEach((l: any) => {
        l.source = nameToIndex.get(l.source);
        l.target = nameToIndex.get(l.target);
      });

      setData({ nodes, links });
    });
  }, []);

  const drawSankey = (width: number, data: SankeyData) => {
    // highlight samples for a given node
    // Global nodeSamplesMap for sharing across highlightSamplesForNode and CSV parsing
    let nodeSamplesMap = new Map<string, string[]>();
    let currentHighlightedNode: string | null = null;

      function highlightSamplesForNode(nodeName: string) {
        const samples = nodeSamplesMap.get(nodeName) || [];
        // Debug: Log which node was clicked and how many samples are associated
        console.log("Node clicked:", nodeName, "Samples found:", samples.length);
        const svgSel = d3.select(svg.node());

        // If toggling off (clicking same node again)
        if (currentHighlightedNode === nodeName) {
          // Debug: Log toggling off node
          console.log("Toggling off node:", nodeName);
          // Restore all main sankey paths to normal color/opacity
          svgSel.selectAll(".links path")
            .transition().duration(300)
            .attr("stroke", function(d: any) {
              // Restore color by layer
              const sourceNode = d.source;
              const sourceLayer = typeof sourceNode === "object" && sourceNode.name
                ? getLayerName(sourceNode.name)
                : "";
              return layerColorMap.get(sourceLayer) ?? "#000";
            })
            .attr("opacity", 0.4);
          svgSel.selectAll(".sample-trajectories line")
            .transition().duration(300)
            .attr("opacity", 0);
          svgSel.selectAll(".sample-points circle")
            .transition().duration(300)
            .attr("opacity", 0);
          currentHighlightedNode = null;
          return;
        }

        currentHighlightedNode = nodeName;

        // Dim all main sankey paths
        svgSel.selectAll(".links path")
          .transition().duration(200)
          .attr("stroke", "#ccc")
          .attr("opacity", 0.2);

        // Hide all sample lines and points
        svgSel.selectAll(".sample-trajectories line")
          .transition().duration(200)
          .attr("opacity", 0);
        svgSel.selectAll(".sample-points circle")
          .transition().duration(200)
          .attr("opacity", 0);

        // Debug: Log number of samples being highlighted
        console.log("Highlighting", samples.length, "samples for node", nodeName);
        // Show only the samples associated with this node
        samples.forEach(sample_id => {
          svgSel.selectAll(`.sample-${sample_id}`)
            .transition().duration(300)
            .attr("opacity", 0.8);
        });
      }

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const height = 600;

    const sankeyGen = sankey<Node, Link>()
      .nodeWidth(5)
      .nodePadding(5)
      .extent([
        [1, 1],
        [width - 1, height - 5],
      ]);

    const { nodes: layoutNodes, links: layoutLinks } = sankeyGen({
      nodes: data.nodes.map((d) => Object.assign({}, d)),
      links: data.links.map((d) => Object.assign({}, d)),
    });

    svg.attr("width", width).attr("height", height);

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
        d3.schemePastel1[i % d3.schemePastel1.length] // ✅ 不加括号，直接取数组元素
      );
    });

    // Sort nodes within each layer by y0 (top to bottom)
    const nodesByLayer = new Map<string, Node[]>();
    layerNames.forEach(layerName => {
      const nodesInLayer = layoutNodes.filter(n => getLayerName(n.name) === layerName);
      nodesInLayer.sort((a, b) => (a.y0 ?? 0) - (b.y0 ?? 0));
      nodesByLayer.set(layerName, nodesInLayer);
    });

    // Draw node lines as thin black vertical lines
    svg
      .append("g")
      .selectAll("line")
      .data(layoutNodes)
      .join("line")
      .attr("x1", (d) => (d.x0 ?? 0) + sankeyGen.nodeWidth()/2)
      .attr("x2", (d) => (d.x0 ?? 0) + sankeyGen.nodeWidth()/2)
      .attr("y1", (d) => d.y0 ?? 0)
      .attr("y2", (d) => d.y1 ?? 0)
      .attr("stroke", "black")
      .attr("stroke-width", sankeyGen.nodeWidth())
      .on("click", (event, d) => highlightSamplesForNode(d.name));

    // Draw aggregated links as colored wide paths
    const linkGroup = svg
      .append("g")
      .attr("fill", "none")
      .attr("class", "links");

    linkGroup
      .selectAll("path")
      .data(layoutLinks)
      .join("path")
      .attr("d", sankeyLinkHorizontal())
      .attr("stroke", (d) => {
        const sourceNode = d.source as Node;
        const sourceLayer = getLayerName(sourceNode.name);
        return layerColorMap.get(sourceLayer) ?? "#000";
      })
      .attr("stroke-width", (d) => Math.max(1, d.width ?? 1))
      .attr("opacity", 0.4);

    // Draw node labels
    svg
      .append("g")
      .selectAll("text")
      .data(layoutNodes)
      .join("text")
      .attr("x", (d) => (d.x0 ?? 0) - 6)
      .attr("y", (d) => ((d.y1 ?? 0) + (d.y0 ?? 0)) / 2)
      .attr("dy", "0.35em")
      .attr("text-anchor", "end")
      .text((d) => d.name)
      .filter((d) => (d.x0 ?? 0) < width / 2)
      .attr("x", (d) => (d.x1 ?? 0) + 6)
      .attr("text-anchor", "start");

    // ----------- New: Draw sample-level points and trajectories -----------

    d3.csv(sampleFile).then(samplePaths => {
      // samplePaths: array of objects with sample_id and layer components

      // Build map: sample_id -> array of node names in layer order
      // layerNames is derived from data.nodes, but to be safe, get from samplePaths keys excluding sample_id
      const pathLayerNames = Object.keys(samplePaths[0]).filter(k => k !== "sample_id");

      // For each sample, build array of node names in order
      const samplePathsMap = new Map<string, string[]>();
      samplePaths.forEach(row => {
        const sample_id = row["sample_id"];
        const nodesInPath = pathLayerNames.map(layer => `${layer}_${row[layer]}`);
        samplePathsMap.set(sample_id, nodesInPath);
      });

      // For each node, collect samples assigned to it
      // Map nodeName -> array of sample_ids
      nodeSamplesMap = new Map<string, string[]>();
      samplePathsMap.forEach((nodeNames, sample_id) => {
        nodeNames.forEach(nodeName => {
          if (!nodeSamplesMap.has(nodeName)) {
            nodeSamplesMap.set(nodeName, []);
          }
          nodeSamplesMap.get(nodeName)!.push(sample_id);
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

      // Map sample_id -> Map nodeName -> y position (for previous layer)
      // We will assign sampleNodeYMap progressively from first layer to last layer
      const sampleNodeYMap = new Map<string, Map<string, number>>();

      // Build map nodeName -> node layout object for quick lookup
      const nodeLayoutMap = new Map<string, Node>();
      layoutNodes.forEach(n => nodeLayoutMap.set(n.name, n));

      // Helper to get downstream sample y positions for a sample at a node
      function getDownstreamY(sample_id: string, nodeName: string): number[] {
        const outgoingNodes = nodeOutgoingMap.get(nodeName);
        if (!outgoingNodes) return [];
        const ys: number[] = [];
        outgoingNodes.forEach(outNodeName => {
          const y = sampleNodeYMap.get(sample_id)?.get(outNodeName);
          if (y !== undefined) ys.push(y);
        });
        return ys;
      }

      // Process layers in reverse order (output-oriented, downstream-based)
      for (let layerIndex = pathLayerNames.length - 1; layerIndex >= 0; layerIndex--) {
        const layerName = pathLayerNames[layerIndex];
        // Get nodes in this layer
        const nodesInLayer = layoutNodes.filter(n => getLayerName(n.name) === layerName);

        nodesInLayer.forEach(node => {
          const samples = nodeSamplesMap.get(node.name) || [];
          if (samples.length === 0) return;

          // For each sample, compute average downstream y position if any
          // Map sample_id -> avg downstream y
          const sampleDownstreamYMap = new Map<string, number | null>();
          samples.forEach(sample_id => {
            const downstreamYs = getDownstreamY(sample_id, node.name);
            if (downstreamYs.length > 0) {
              const avgY = d3.mean(downstreamYs) ?? null;
              sampleDownstreamYMap.set(sample_id, avgY);
            } else {
              sampleDownstreamYMap.set(sample_id, null);
            }
          });

          // Sort samples: those with downstream info by avg downstream y ascending,
          // those without downstream info by numeric sample_id ascending
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
              // fallback to numeric sample_id sort
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
          samples.forEach((sample_id, i) => {
            const y = count === 1 ? y0 + height / 2 : y0 + (i + 0.5) * (height / count);
            if (!sampleNodeYMap.has(sample_id)) {
              sampleNodeYMap.set(sample_id, new Map());
            }
            sampleNodeYMap.get(sample_id)!.set(node.name, y);
          });
        });
      }

      // Prepare group for sample trajectories (per-link lines)
      const samplesGroup = svg.append("g").attr("class", "sample-trajectories");

      // Build a map from link key to sample IDs that traverse that link
      // link key format: sourceNodeName->targetNodeName (matches layoutLinks key)
      const linkSamplesMap = new Map<string, string[]>();

      samplePathsMap.forEach((nodeNames, sample_id) => {
        for (let i = 0; i < nodeNames.length - 1; i++) {
          const sourceNodeName = nodeNames[i];
          const targetNodeName = nodeNames[i + 1];
          const linkKey = `${sourceNodeName}->${targetNodeName}`;
          if (!linkSamplesMap.has(linkKey)) {
            linkSamplesMap.set(linkKey, []);
          }
          linkSamplesMap.get(linkKey)!.push(sample_id);
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

        const sourceLayer = getLayerName(sourceNode.name);
        const strokeColor = layerColorMap.get(sourceLayer) ?? "#999";

        samplesOnLink.forEach(sample_id => {
          const ySource = sampleNodeYMap.get(sample_id)?.get(sourceNode.name);
          const yTarget = sampleNodeYMap.get(sample_id)?.get(targetNode.name);
          if (ySource === undefined || yTarget === undefined) return;

          const xSource = (sourceNode.x0 ?? 0) + sankeyGen.nodeWidth()/2;
          const xTarget = (targetNode.x0 ?? 0) + sankeyGen.nodeWidth()/2;

          // Compose CSS class: link key plus sample IDs
          // E.g., class="link-A_B sample-3"
          // To keep class names simple, replace '->' with '-' and remove underscores
          const safeLinkClass = `link-${linkKey.replace(/->/g, "-").replace(/_/g, "_")}`;
          const sampleClass = `sample-${sample_id}`;
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
        .attr("class", d => d.className);

      // Draw sample points as small circles
      const samplePointsGroup = svg.append("g").attr("class", "sample-points");

      sampleNodeYMap.forEach((nodeYMap, sample_id) => {
        nodeYMap.forEach((y, nodeName) => {
          const node = nodeLayoutMap.get(nodeName);
          if (!node) return;
          const x = (node.x0 ?? 0) + sankeyGen.nodeWidth()/2;
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
    <div ref={containerRef} style={{ width: "100%", height: "600px", position: "relative" }}>
      <svg ref={ref}></svg>
    </div>
  );
}