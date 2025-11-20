# Molecule Images

This directory contains molecule structure images generated from SMILES strings in the CSV files.

## File Structure

- `{real_id}_primary.png` - Starting material molecule structure
- `{real_id}_product.png` - Product molecule structure  
- `{real_id}_ligand.png` - Ligand molecule structure
- `{real_id}_solvent.png` - Solvent molecule structure
- `{real_id}_catalyst.png` - Catalyst molecule structure (if available)

## Usage in Frontend

### Direct Image Access
```typescript
const realId = 2348;
const imagePath = `/data/molecule_images/${realId}_primary.png`;
```

### Using the Mapping File
```typescript
import moleculeMapping from '/data/molecule_images/molecule_image_mapping.json';

const realId = 2348;
const imagePath = `/data/molecule_images/${moleculeMapping[realId].primary}`;
```

### Using the Summary CSV
```typescript
// Load the CSV and create a lookup map
const summary = await fetch('/data/molecule_images/molecule_image_summary.csv')
  .then(r => r.text())
  .then(csv => parseCSV(csv));

const imagePath = `/data/molecule_images/${summary.find(r => r.real_id === 2348).primary_image}`;
```

## Image Types

- **primary**: Starting material (always available)
- **product**: Product molecule (always available)
- **ligand**: Ligand molecule (always available)
- **solvent**: Solvent molecule (always available)
- **catalyst**: Catalyst molecule (only available for some reactions)

## Statistics

- Total unique IDs: 380
- Primary images: 380
- Product images: 380
- Ligand images: 380
- Solvent images: 380
- Catalyst images: 18
