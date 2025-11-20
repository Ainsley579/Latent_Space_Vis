// ==========================================
// sankey_core.js
// Pure JS version of the full-featured Sankey
// ==========================================

function renderSankey(container, nodesRaw, linksRaw, sampleRows, splitMapRaw) {
  container.innerHTML = ""; // clear

  const width = container.clientWidth;
  const height = 600;

  // convert splitMapRaw -> Map
  const splitLabelMap = new Map(Object.entries(splitMapRaw || {}));

  // Convert data
  const nodes = nodesRaw.map(d => ({ ...d }));
  const links = linksRaw.map(d => ({ ...d }));

  // Build name→index map
  const nameToIndex = new Map(nodes.map((d, i) => [d.name, i]));
  links.forEach(l => {
    l.source = nameToIndex.get(l.source);
    l.target = nameToIndex.get(l.target);
  });

  // SVG
  const svg = d3.select(container)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  // Sankey
  const sankeyGen = d3.sankey()
    .nodeWidth(10)
    .nodePadding(7)
    .extent([[1, 1], [width - 20, height - 5]]);

  const { nodes: layoutNodes, links: layoutLinks } = sankeyGen({
    nodes: nodes.map(d => ({ ...d })),
    links: links.map(d => ({ ...d }))
  });

  // Helper
  function getLayerName(name) {
    const idx = name.lastIndexOf("_");
    return idx === -1 ? name : name.slice(0, idx);
  }

  // Build samples → path
  const samplePathsMap = new Map();
  sampleRows.forEach(r => {
    const sid = String(r.real_id);
    const fromNode = `${r.from_layer}_${r.from_cluster}`;
    const toNode = `${r.to_layer}_${r.to_cluster}`;

    if (!samplePathsMap.has(sid)) samplePathsMap.set(sid, []);
    const arr = samplePathsMap.get(sid);
    if (arr[arr.length - 1] !== fromNode) arr.push(fromNode);
    arr.push(toNode);
  });

  // Build nodeSamplesMap
  const nodeSamplesMap = new Map();
  samplePathsMap.forEach((path, sid) => {
    path.forEach(n => {
      if (!nodeSamplesMap.has(n)) nodeSamplesMap.set(n, []);
      nodeSamplesMap.get(n).push(sid);
    });
  });

  // ---- Build link → sample IDs ----
  const linkSamplesMap = new Map();
  layoutLinks.forEach(link => {
    const key = `${link.source.name}->${link.target.name}`;
    linkSamplesMap.set(key, []);
  });

  samplePathsMap.forEach((path, sid) => {
    for (let i = 0; i < path.length - 1; i++) {
      const src = path[i];
      const tgt = path[i + 1];
      const key = `${src}->${tgt}`;
      if (linkSamplesMap.has(key)) linkSamplesMap.get(key).push(sid);
    }
  });

  // ================
  // Split Link Drawing
  // ================

  const splits = ["train", "test", "experimental", "missing"];
  const splitColors = {
    train: "#1f77b4",
    test: "#ff7f0e",
    experimental: "#2ca02c",
    missing: "#999999",
  };

  const linkGroup = svg.append("g").attr("class", "sankey-links");

  const separatedLinks = [];

  layoutLinks.forEach(link => {
    const srcNode = link.source;
    const tgtNode = link.target;
    const key = `${srcNode.name}->${tgtNode.name}`;
    const sampleIds = linkSamplesMap.get(key) || [];

    const counts = { train: 0, test: 0, experimental: 0, missing: 0 };
    sampleIds.forEach(id => {
      const lab = splitLabelMap.get(String(id)) || "missing";
      if (counts[lab] !== undefined) counts[lab] += 1;
      else counts.missing += 1;
    });

    const totalCount = counts.train + counts.test + counts.experimental + counts.missing;
    if (totalCount === 0) return;

    const y0 = link.y0;
    const y1 = link.y1;
    const linkWidth = link.width;

    const sx = srcNode.x1;
    const tx = tgtNode.x0;

    const baseOffset = -linkWidth / 2;
    let acc = 0;

    splits.forEach(sp => {
      const cnt = counts[sp];
      if (cnt <= 0) return;

      const segW = (linkWidth * cnt) / totalCount;
      const centerOff = baseOffset + acc + segW / 2;

      const sy = y0 + centerOff;
      const ty = y1 + centerOff;

      separatedLinks.push({
        sx, sy, tx, ty,
        width: segW,
        stroke: splitColors[sp]
      });

      acc += segW;
    });
  });

  // Draw split paths
  linkGroup.selectAll("path")
    .data(separatedLinks)
    .join("path")
    .attr("d", d => {
      const mx = (d.sx + d.tx) / 2;
      return `M${d.sx},${d.sy} C${mx},${d.sy} ${mx},${d.ty} ${d.tx},${d.ty}`;
    })
    .attr("stroke", d => d.stroke)
    .attr("stroke-width", d => d.width)
    .attr("fill", "none")
    .attr("opacity", 0.4);

  // ==============================
  // Node Multi-color Stack
  // ==============================
  const nodeGroup = svg.append("g").attr("class", "node-stacks");

  nodeGroup.selectAll("g.node")
    .data(layoutNodes)
    .join("g")
    .attr("class", "node")
    .each(function (nodeData) {
      const g = d3.select(this);

      const sampleIds = nodeSamplesMap.get(nodeData.name) || [];
      const counts = { train: 0, test: 0, experimental: 0, missing: 0 };

      sampleIds.forEach(id => {
        const label = splitLabelMap.get(String(id)) || "missing";
        counts[label] += 1;
      });

      const total = counts.train + counts.test + counts.experimental + counts.missing;
      const nodeHeight = nodeData.y1 - nodeData.y0;
      const nodeX = nodeData.x0;
      const nodeW = sankeyGen.nodeWidth();

      if (total === 0) {
        g.append("rect")
          .attr("x", nodeX)
          .attr("y", nodeData.y0)
          .attr("width", nodeW)
          .attr("height", nodeHeight)
          .attr("fill", "#ccc");
        // --- Node hover + click highlight ---
        const gsel = g;
        gsel.style("cursor", "pointer")
          .on("mouseover", function () {
            d3.select(this).selectAll("rect")
              .attr("stroke", "black")
              .attr("stroke-width", 2);
          })
          .on("mouseout", function () {
            d3.select(this).selectAll("rect")
              .attr("stroke", "none");
          })
          .on("click", function () {
            const nodeName = nodeData.name;
            const sampleIds = nodeSamplesMap.get(nodeName) || [];

            // If clicked twice → clear
            if (window.__ACTIVE_NODE__ === nodeName) {
              window.__ACTIVE_NODE__ = null;

              svg.selectAll(".sankey-links path")
                .attr("opacity", 0.4)
                .attr("stroke", d => d.stroke);

              svg.selectAll(".sample-trajectories line")
                .attr("opacity", 0)
                .attr("stroke-width", 0.5);

              window.dispatchEvent(new CustomEvent("sankey-highlight-clear"));
              return;
            }

            // Set active
            window.__ACTIVE_NODE__ = nodeName;

            // Dim everything
            svg.selectAll(".sankey-links path").attr("opacity", 0.1);
            svg.selectAll(".sample-trajectories line").attr("opacity", 0.05);

            // Highlight samples of this node
            sampleIds.forEach(id => {
              svg.selectAll(`.sample-${id}`)
                .attr("opacity", 1)
                .attr("stroke-width", 1.8);
            });

            // Broadcast for scatter
            window.dispatchEvent(new CustomEvent("sankey-highlight-ids", { detail: sampleIds }));
          });
        return;
      }

      let accY = nodeData.y0;
      splits.forEach(sp => {
        const c = counts[sp];
        if (c <= 0) return;
        const h = (c / total) * nodeHeight;

        g.append("rect")
          .attr("x", nodeX)
          .attr("y", accY)
          .attr("width", nodeW)
          .attr("height", h)
          .attr("fill", splitColors[sp]);

        accY += h;
      });
      // --- Node hover + click highlight ---
      const gsel = g;
      gsel.style("cursor", "pointer")
        .on("mouseover", function () {
          d3.select(this).selectAll("rect")
            .attr("stroke", "black")
            .attr("stroke-width", 2);
        })
        .on("mouseout", function () {
          d3.select(this).selectAll("rect")
            .attr("stroke", "none");
        })
        .on("click", function () {
          const nodeName = nodeData.name;
          const sampleIds = nodeSamplesMap.get(nodeName) || [];

          // If clicked twice → clear
          if (window.__ACTIVE_NODE__ === nodeName) {
            window.__ACTIVE_NODE__ = null;

            svg.selectAll(".sankey-links path")
              .attr("opacity", 0.4)
              .attr("stroke", d => d.stroke);

            svg.selectAll(".sample-trajectories line")
              .attr("opacity", 0)
              .attr("stroke-width", 0.5);

            window.dispatchEvent(new CustomEvent("sankey-highlight-clear"));
            return;
          }

          // Set active
          window.__ACTIVE_NODE__ = nodeName;

          // Dim everything
          svg.selectAll(".sankey-links path").attr("opacity", 0.1);
          svg.selectAll(".sample-trajectories line").attr("opacity", 0.05);

          // Highlight samples of this node
          sampleIds.forEach(id => {
            svg.selectAll(`.sample-${id}`)
              .attr("opacity", 1)
              .attr("stroke-width", 1.8);
          });

          // Broadcast for scatter
          window.dispatchEvent(new CustomEvent("sankey-highlight-ids", { detail: sampleIds }));
        });
    });

  // ============================
  // Sample Trajectories
  // ============================

  const nodeLayoutMap = new Map();
  layoutNodes.forEach(n => nodeLayoutMap.set(n.name, n));

  // compute y positions per sample (simple evenly spaced)
  const sampleNodeYMap = new Map();

  layoutNodes.forEach(node => {
    const samples = nodeSamplesMap.get(node.name) || [];
    const count = samples.length;
    const y0 = node.y0;
    const h = node.y1 - node.y0;

    samples.forEach((sid, i) => {
      const y = y0 + (i + 0.5) * (h / count);
      if (!sampleNodeYMap.has(sid)) sampleNodeYMap.set(sid, new Map());
      sampleNodeYMap.get(sid).set(node.name, y);
    });
  });

  const trajectories = svg.append("g").attr("class", "sample-trajectories");

  const lineData = [];
  layoutLinks.forEach(link => {
    const src = link.source;
    const tgt = link.target;
    const key = `${src.name}->${tgt.name}`;
    const sidArr = linkSamplesMap.get(key) || [];

    sidArr.forEach(sid => {
      const y1 = sampleNodeYMap.get(sid)?.get(src.name);
      const y2 = sampleNodeYMap.get(sid)?.get(tgt.name);
      if (y1 == null || y2 == null) return;

      const x1 = src.x0 + sankeyGen.nodeWidth() / 2;
      const x2 = tgt.x0 + sankeyGen.nodeWidth() / 2;

      let c = "#999";
      const raw = splitLabelMap.get(sid);
      if (raw === "train") c = "#1f77b4";
      else if (raw === "test") c = "#ff7f0e";
      else if (raw === "experimental") c = "#2ca02c";

      lineData.push({ x1, y1, x2, y2, sid, stroke: c });
    });
  });

  trajectories.selectAll("line")
    .data(lineData)
    .join("line")
    .attr("x1", d => d.x1)
    .attr("y1", d => d.y1)
    .attr("x2", d => d.x2)
    .attr("y2", d => d.y2)
    .attr("stroke", d => d.stroke)
    .attr("stroke-width", 0.5)
    .attr("opacity", 0)
    .attr("class", d => `sample-${d.sid}`)
    .on("mouseover", function (ev, d) {
      const tooltip = d3.select("body").append("div")
        .style("position", "absolute")
        .style("padding", "4px")
        .style("background", "white")
        .style("border", "1px solid #ccc")
        .style("border-radius", "4px")
        .style("pointer-events", "none")
        .style("z-index", 9999)
        .style("opacity", 1);

      tooltip
        .html(`<img src="/data/molecule_images/${d.sid}_ligand.png" width="180"/>`)
        .style("left", (ev.pageX + 12) + "px")
        .style("top", (ev.pageY + 12) + "px");

      d3.select(this).on("mouseout", () => tooltip.remove());
    });

  // ============================
  // Highlight Events
  // ============================

  window.addEventListener("sankey-highlight-ids", e => {
    const ids = e.detail || [];
    svg.selectAll(".sample-trajectories line")
      .attr("opacity", 0.05);

    ids.forEach(id => {
      svg.selectAll(`.sample-${id}`)
        .attr("opacity", 1)
        .attr("stroke-width", 1.8);
    });
  });

  window.addEventListener("sankey-highlight-clear", () => {
    svg.selectAll(".sample-trajectories line")
      .attr("opacity", 0)
      .attr("stroke-width", 0.5);

    svg.selectAll(".sankey-links path")
      .attr("opacity", 0.4)
      .attr("stroke", d => d.stroke);
  });
}