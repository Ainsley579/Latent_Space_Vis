# Frontend Usage: Distinguishing All Data Points

We have **3,000+ data points** but only **4 unique products**. Use **unique identifiers** that combine reaction (rxn_id) + conformer (catalyst_key) information.

## Quick Start

### 1. Load the Lookup Table

```typescript
// Load once at app start
const lookupResponse = await fetch('/data/unique_identifiers/frontend_lookup.json');
const lookup = await lookupResponse.json();
```

### 2. Get Unique Information for Each Data Point

```typescript
// For any real_id, get unique identifier
const realId = 2348;
const data = lookup.by_real_id[realId];

console.log(data.unique_id);        // "HF-1_1879897-28-5-conformer-1"
console.log(data.display_name);     // "HF-1 (1)"
console.log(data.rxn_id);           // "HF-1"
console.log(data.catalyst_key);      // "1879897-28-5_conformer-1"
console.log(data.ddG);              // 2.066
```

## Visual Distinction Strategies

### Strategy 1: Show Unique Identifier in Tooltip/Label

```typescript
const DataPoint = ({ realId, x, y }) => {
  const data = lookup.by_real_id[realId];
  
  return (
    <g>
      <circle cx={x} cy={y} r={5} />
      <Tooltip>
        <div>
          <div><strong>Reaction:</strong> {data.rxn_id}</div>
          <div><strong>Conformer:</strong> {data.catalyst_key}</div>
          <div><strong>Unique ID:</strong> {data.unique_id}</div>
          <div><strong>ddG:</strong> {data.ddG}</div>
          <img src={`/data/molecule_images/${realId}_primary.png`} />
        </div>
      </Tooltip>
    </g>
  );
};
```

### Strategy 2: Color-Code by Unique Combination

```typescript
// Create color map based on unique_id
const colorMap = new Map();
let colorIndex = 0;
Object.values(lookup.by_real_id).forEach(data => {
  if (!colorMap.has(data.unique_id)) {
    colorMap.set(data.unique_id, getColor(colorIndex++));
  }
});

const DataPoint = ({ realId, x, y }) => {
  const data = lookup.by_real_id[realId];
  const color = colorMap.get(data.unique_id);
  
  return (
    <circle 
      cx={x} 
      cy={y} 
      r={5} 
      fill={color}
      stroke="black"
      strokeWidth={1}
    />
  );
};
```

### Strategy 3: Show Multiple Molecules Per Point

```typescript
const DataPointWithMolecules = ({ realId, x, y }) => {
  const data = lookup.by_real_id[realId];
  
  return (
    <g>
      <circle cx={x} cy={y} r={5} />
      <Tooltip>
        <div className="molecule-comparison">
          <div>
            <div>Product (Reaction Output)</div>
            <img src={`/data/molecule_images/${realId}_primary.png`} />
          </div>
          <div>
            <div>Ligand (Conformer Variation)</div>
            <img src={`/data/molecule_images/${realId}_ligand.png`} />
          </div>
          <div className="info">
            <div>Reaction: {data.rxn_id}</div>
            <div>Conformer: {data.catalyst_key}</div>
            <div>ddG: {data.ddG}</div>
          </div>
        </div>
      </Tooltip>
    </g>
  );
};
```

### Strategy 4: Use Unique ID as Text Label

```typescript
const DataPointWithLabel = ({ realId, x, y }) => {
  const data = lookup.by_real_id[realId];
  
  return (
    <g>
      <circle cx={x} cy={y} r={5} />
      <text 
        x={x + 8} 
        y={y} 
        fontSize="10px"
        fill="black"
      >
        {data.display_name}
      </text>
    </g>
  );
};
```

### Strategy 5: Group by Reaction, Show Conformers

```typescript
// Group data points by reaction
const groupByReaction = (dataPoints) => {
  const grouped = {};
  dataPoints.forEach(dp => {
    const data = lookup.by_real_id[dp.realId];
    if (!grouped[data.rxn_id]) {
      grouped[data.rxn_id] = [];
    }
    grouped[data.rxn_id].push(dp);
  });
  return grouped;
};

// Display with reaction label and conformer count
const ReactionGroup = ({ rxnId, dataPoints }) => {
  const conformers = lookup.by_rxn_id[rxnId];
  
  return (
    <div className="reaction-group">
      <div className="reaction-header">
        <strong>{rxnId}</strong> ({conformers.length} conformers)
      </div>
      <div className="conformers">
        {conformers.map(conf => (
          <div key={conf.real_id} className="conformer">
            <img src={`/data/molecule_images/${conf.real_id}_primary.png`} />
            <div>Conformer: {conf.catalyst_key}</div>
            <div>ddG: {conf.ddG}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
```

## Complete Example Component

```typescript
import React, { useState, useEffect } from 'react';

const DataPointVisualization = ({ realId, x, y }) => {
  const [lookup, setLookup] = useState(null);
  const [hovered, setHovered] = useState(false);
  
  useEffect(() => {
    fetch('/data/unique_identifiers/frontend_lookup.json')
      .then(r => r.json())
      .then(data => setLookup(data));
  }, []);
  
  if (!lookup) return null;
  
  const data = lookup.by_real_id[realId];
  if (!data) return null;
  
  // Color based on unique_id hash
  const color = `hsl(${(data.unique_id.length * 17) % 360}, 70%, 50%)`;
  
  return (
    <g 
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <circle 
        cx={x} 
        cy={y} 
        r={hovered ? 8 : 5}
        fill={color}
        stroke="black"
        strokeWidth={hovered ? 2 : 1}
        style={{ cursor: 'pointer' }}
      />
      
      {hovered && (
        <foreignObject x={x + 15} y={y - 50} width="300" height="200">
          <div className="tooltip" style={{
            background: 'white',
            border: '1px solid #ccc',
            padding: '10px',
            borderRadius: '5px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
          }}>
            <div><strong>Reaction:</strong> {data.rxn_id}</div>
            <div><strong>Conformer:</strong> {data.catalyst_key}</div>
            <div><strong>Unique ID:</strong> {data.unique_id}</div>
            <div><strong>ddG:</strong> {data.ddG}</div>
            <div><strong>Split:</strong> {data.split_label}</div>
            <img 
              src={`/data/molecule_images/${realId}_primary.png`}
              style={{ width: '150px', marginTop: '10px' }}
              alt="Molecule structure"
            />
          </div>
        </foreignObject>
      )}
    </g>
  );
};
```

## Lookup Methods

### By Real ID (Most Common)
```typescript
const data = lookup.by_real_id[realId];
```

### By Unique ID
```typescript
const data = lookup.by_unique_id['HF-1_1879897-28-5-conformer-1'];
```

### By Reaction ID (Grouped)
```typescript
const conformers = lookup.by_rxn_id['HF-1'];
// Returns array of all data points for this reaction
```

## Data Structure

Each entry contains:
- `unique_id`: Combined identifier (rxn_id + catalyst_key)
- `display_name`: Human-readable name (e.g., "HF-1 (1)")
- `rxn_id`: Reaction identifier (e.g., "HF-1")
- `catalyst_key`: Conformer identifier (e.g., "1879897-28-5_conformer-1")
- `ddG`: Reaction outcome value
- `split_label`: Dataset split (train/test/experimental)
- `startingmat_smiles`: Starting material SMILES
- `product_smiles`: Product SMILES
- `ligand_smiles`: Ligand SMILES

## Summary

✅ **Each data point now has a unique identifier** combining reaction + conformer
✅ **380 unique data points** (not just 4 products)
✅ **Multiple lookup methods** for easy access
✅ **Complete information** for tooltips and labels

Use the `unique_id` or `display_name` to distinguish each data point visually!

