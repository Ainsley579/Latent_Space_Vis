# Which Molecule Type Should You Use?

## Quick Answer

**For most visualizations, use `product` (now the default "primary")**

The `primary.png` images have been regenerated to show **product molecules** instead of starting materials.

## Molecule Type Comparison

| Molecule Type | Unique Count | Diversity | Best For |
|--------------|--------------|-----------|----------|
| **Product** | 4 unique | Low | ✅ **Default choice** - Shows reaction output |
| **Starting Material** | 4 unique | Very Low | ❌ Not recommended - Most reactions use same input |
| **Ligand** | 153 unique | **High** | ✅ Maximum diversity - Shows catalyst variation |
| **Solvent** | 3 unique | Very Low | ❌ Not recommended - Limited variation |
| **Catalyst** | 5 unique | Low | ⚠️ Only available for some reactions |

## Recommendations by Use Case

### 1. **General Visualization (Recommended)**
```typescript
// Use PRODUCT - shows what each reaction produces
<img src={`/data/molecule_images/${realId}_primary.png`} />
// OR explicitly:
<img src={`/data/molecule_images/${realId}_product.png`} />
```
**Why:** Product molecules represent the reaction outcome, which is what you're analyzing.

### 2. **Maximum Visual Diversity**
```typescript
// Use LIGAND - 153 unique structures
<img src={`/data/molecule_images/${realId}_ligand.png`} />
```
**Why:** Ligands have the highest diversity (153 unique vs 4 unique products). Use this if you want to see the most variation in molecule structures.

### 3. **Show Reaction Components**
```typescript
// Display all components side-by-side
<div>
  <img src={`/data/molecule_images/${realId}_startingmat.png`} alt="Starting Material" />
  <img src={`/data/molecule_images/${realId}_product.png`} alt="Product" />
  <img src={`/data/molecule_images/${realId}_ligand.png`} alt="Ligand" />
</div>
```
**Why:** Shows the full reaction context.

## Understanding the Data

Your dataset contains **hydroformylation reactions** where:
- **Starting materials** are mostly the same (1-octene, 1-heptene, etc.)
- **Products** are aldehydes (linear and branched)
- **Ligands** vary significantly (153 different ligands)
- **Solvents** are mostly toluene

This is why:
- Starting materials look the same (only 4 unique)
- Products also have limited diversity (4 unique)
- **Ligands show the most variation** (153 unique)

## Updated Primary Images

The `{realId}_primary.png` files have been regenerated to show **product molecules** instead of starting materials. This is now the recommended default.

## Example Usage

```typescript
// Option 1: Use primary (now shows product)
const MoleculeImage = ({ realId }) => (
  <img src={`/data/molecule_images/${realId}_primary.png`} />
);

// Option 2: Use product explicitly
const ProductImage = ({ realId }) => (
  <img src={`/data/molecule_images/${realId}_product.png`} />
);

// Option 3: Use ligand for maximum diversity
const LigandImage = ({ realId }) => (
  <img src={`/data/molecule_images/${realId}_ligand.png`} />
);

// Option 4: Show all components
const AllMolecules = ({ realId }) => (
  <div className="molecule-grid">
    <div>
      <label>Starting Material</label>
      <img src={`/data/molecule_images/${realId}_startingmat.png`} />
    </div>
    <div>
      <label>Product</label>
      <img src={`/data/molecule_images/${realId}_product.png`} />
    </div>
    <div>
      <label>Ligand</label>
      <img src={`/data/molecule_images/${realId}_ligand.png`} />
    </div>
  </div>
);
```

## Summary

- ✅ **Use `primary` or `product`** for general visualization (default)
- ✅ **Use `ligand`** if you want maximum visual diversity
- ⚠️ **Avoid `startingmat`** for visualization (too repetitive)
- ⚠️ **Avoid `solvent`** for visualization (too repetitive)

