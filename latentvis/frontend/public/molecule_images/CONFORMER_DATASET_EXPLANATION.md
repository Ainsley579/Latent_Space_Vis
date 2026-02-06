# Why So Few Unique Molecules?

## The Question

You have **~3,000 data points** but only **4 unique products** and **4 unique starting materials**. Why?

## The Answer: This is a Conformer Dataset

Your dataset is from **`dimenet-10conf`** (10 conformers), which means:

### Data Structure

- **3,518 total entries** in the original dataset
- **~324 unique reactions** (rxn_id)
- **~1,311 unique conformers** (catalyst_key)
- **~10 conformers per reaction** (on average)

### What's a Conformer?

A **conformer** is a different 3D structure of the **same molecule**:
- Same **2D structure** (same SMILES string)
- Different **3D geometry** (different spatial arrangement)
- Different **reaction outcomes** (different ddG values)

### Example

```
Reaction: HF-1
├── conformer-1 → ddG = 2.41
├── conformer-2 → ddG = 2.35
├── conformer-3 → ddG = 2.50
└── conformer-4 → ddG = 2.28
```

All have:
- ✅ Same starting material SMILES: `C=CCCCCCC`
- ✅ Same product SMILES: `CCCCCCCCC([H])=O`
- ✅ Same ligand SMILES: `[H]C1=CC([H])=C(OP2OC3=...`
- ❌ **Different 3D structures** (different conformers)
- ❌ **Different ddG values** (different reaction outcomes)

## Why SMILES Look the Same

**SMILES only represents 2D molecular structure**, not 3D:

- Conformers have the **same 2D structure** → same SMILES
- But **different 3D structures** → different conformations
- The 3D difference affects reaction outcome (ddG) but **not the SMILES string**

## Why Only 4 Unique Products?

Your dataset contains **hydroformylation reactions**:
- Most reactions use similar starting materials (olefins: 1-octene, 1-heptene, etc.)
- Most produce similar products (aldehydes: linear and branched)
- Only a few different chain lengths: C8, C7, C6, C5

So across all ~324 reactions, you only have:
- **4 unique starting materials** (different chain lengths)
- **4 unique products** (corresponding aldehydes)
- **153 unique ligands** (different ligand structures)
- **1,311 unique conformers** (different 3D structures)

## Visualization Strategies

### Option 1: Group by Reaction (rxn_id)
```typescript
// Show one point per unique reaction
const reactions = data.groupBy('rxn_id');
// Each reaction has unique molecules
```

**Pros:**
- Shows true reaction diversity
- Each point has unique molecules

**Cons:**
- Loses conformer information
- Reduces from 3,518 to ~324 points

### Option 2: Show Conformer Info
```typescript
// Display conformer information alongside molecule
const MoleculeWithConformer = ({ realId, rxnId, catalystKey }) => (
  <div>
    <img src={`/data/molecule_images/${realId}_product.png`} />
    <div>Reaction: {rxnId}</div>
    <div>Conformer: {catalystKey}</div>
  </div>
);
```

**Pros:**
- Shows all data points
- Includes conformer information

**Cons:**
- Molecules look similar (same SMILES)
- Need to show conformer info separately

### Option 3: Use Ligand for Diversity
```typescript
// Use ligand images (153 unique) for visual diversity
<img src={`/data/molecule_images/${realId}_ligand.png`} />
```

**Pros:**
- Maximum visual diversity (153 unique)
- Shows catalyst/ligand variation

**Cons:**
- Doesn't show reaction product
- Ligands are complex and may look similar

### Option 4: Color by Conformer
```typescript
// Use product molecule but color-code by conformer
const color = getConformerColor(catalystKey);
<img src={`/data/molecule_images/${realId}_product.png`} style={{border: `3px solid ${color}`}} />
```

**Pros:**
- Shows reaction product
- Distinguishes conformers visually

**Cons:**
- Still same molecule structure
- Need color legend

## Recommended Approach

For your visualization, I recommend:

1. **Use product molecules** (`primary.png`) - shows reaction outcome
2. **Display conformer info** in tooltip/hover - shows 3D structure difference
3. **Color-code by ddG** - shows reaction outcome variation
4. **Group by rxn_id** if you want unique molecules per point

```typescript
const DataPoint = ({ realId, rxnId, catalystKey, ddG }) => (
  <div className="data-point" style={{borderColor: getColorByDDG(ddG)}}>
    <img src={`/data/molecule_images/${realId}_primary.png`} />
    <Tooltip>
      <div>Reaction: {rxnId}</div>
      <div>Conformer: {catalystKey}</div>
      <div>ddG: {ddG}</div>
    </Tooltip>
  </div>
);
```

## Summary

- ✅ **3,518 data points** = 324 reactions × ~10 conformers each
- ✅ **4 unique products** = only a few different aldehydes across all reactions
- ✅ **Same SMILES** = conformers have same 2D structure
- ✅ **Different ddG** = conformers have different 3D structures and outcomes

The molecules look the same because **SMILES only captures 2D structure**, but your data points differ in **3D conformer geometry**, which affects reaction outcomes.

