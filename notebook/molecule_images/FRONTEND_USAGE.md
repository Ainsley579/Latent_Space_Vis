# Frontend Usage Guide for Molecule Images

## Quick Start

### 1. Direct Image Access (Simplest)

```typescript
// In your React/Next.js component
const MoleculeImage = ({ realId, moleculeType = 'primary' }) => {
  const imagePath = `/data/molecule_images/${realId}_${moleculeType}.png`;
  
  return (
    <img 
      src={imagePath} 
      alt={`Molecule ${realId} - ${moleculeType}`}
      style={{ width: '400px', height: '400px' }}
      onError={(e) => {
        // Handle missing images gracefully
        e.target.style.display = 'none';
      }}
    />
  );
};

// Usage
<MoleculeImage realId={2348} moleculeType="primary" />
<MoleculeImage realId={2348} moleculeType="product" />
```

### 2. Using the Mapping JSON File

```typescript
// Load mapping once (e.g., in a useEffect or at component mount)
import moleculeMapping from '/data/molecule_images/molecule_image_mapping.json';

const MoleculeImageWithMapping = ({ realId, moleculeType = 'primary' }) => {
  const imagePath = moleculeMapping[realId]?.[moleculeType] 
    ? `/data/molecule_images/${moleculeMapping[realId][moleculeType]}`
    : null;
  
  if (!imagePath) {
    return <div>Image not available</div>;
  }
  
  return <img src={imagePath} alt={`Molecule ${realId}`} />;
};
```

### 3. Using the Summary CSV

```typescript
// Load and parse CSV
const loadMoleculeSummary = async () => {
  const response = await fetch('/data/molecule_images/molecule_image_summary.csv');
  const csvText = await response.text();
  
  // Simple CSV parser (or use a library like papaparse)
  const lines = csvText.split('\n');
  const headers = lines[0].split(',');
  const data = lines.slice(1).map(line => {
    const values = line.split(',');
    return headers.reduce((obj, header, i) => {
      obj[header] = values[i];
      return obj;
    }, {});
  });
  
  return data;
};

// Create lookup map
const moleculeSummary = await loadMoleculeSummary();
const imageMap = new Map(
  moleculeSummary.map(row => [row.real_id, row])
);

// Use in component
const MoleculeImageFromCSV = ({ realId, moleculeType = 'primary' }) => {
  const row = imageMap.get(realId.toString());
  const imageFile = row?.[`${moleculeType}_image`];
  
  if (!imageFile) return null;
  
  return <img src={`/data/molecule_images/${imageFile}`} alt={`Molecule ${realId}`} />;
};
```

## Available Molecule Types

- `primary` - Starting material (always available)
- `product` - Product molecule (always available)
- `ligand` - Ligand molecule (always available)
- `solvent` - Solvent molecule (always available)
- `catalyst` - Catalyst molecule (only for some reactions, check if exists)

## Example: Display All Molecules for a Data Point

```typescript
const AllMoleculesView = ({ realId }) => {
  const moleculeTypes = ['primary', 'product', 'ligand', 'solvent', 'catalyst'];
  
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
      {moleculeTypes.map(type => (
        <div key={type} style={{ textAlign: 'center' }}>
          <img 
            src={`/data/molecule_images/${realId}_${type}.png`}
            alt={`${type} molecule`}
            style={{ width: '200px', height: '200px' }}
            onError={(e) => e.target.style.display = 'none'}
          />
          <div style={{ fontSize: '12px', marginTop: '5px' }}>{type}</div>
        </div>
      ))}
    </div>
  );
};
```

## Example: Tooltip with Molecule Structure

```typescript
const DataPointWithTooltip = ({ realId, x, y }) => {
  const [showTooltip, setShowTooltip] = useState(false);
  
  return (
    <div 
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
      style={{ position: 'relative' }}
    >
      <circle cx={x} cy={y} r={5} />
      
      {showTooltip && (
        <div style={{
          position: 'absolute',
          top: y - 250,
          left: x + 10,
          background: 'white',
          border: '1px solid #ccc',
          padding: '10px',
          zIndex: 1000,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
        }}>
          <img 
            src={`/data/molecule_images/${realId}_primary.png`}
            alt={`Molecule ${realId}`}
            style={{ width: '200px', height: '200px' }}
          />
          <div style={{ fontSize: '12px', marginTop: '5px' }}>
            ID: {realId}
          </div>
        </div>
      )}
    </div>
  );
};
```

## Error Handling

Always handle cases where images might not exist:

```typescript
const SafeMoleculeImage = ({ realId, moleculeType = 'primary' }) => {
  const [imageError, setImageError] = useState(false);
  const imagePath = `/data/molecule_images/${realId}_${moleculeType}.png`;
  
  if (imageError) {
    return <div style={{ width: '200px', height: '200px', border: '1px dashed #ccc' }}>
      Image not available
    </div>;
  }
  
  return (
    <img 
      src={imagePath}
      alt={`Molecule ${realId}`}
      onError={() => setImageError(true)}
      style={{ width: '200px', height: '200px' }}
    />
  );
};
```

## Performance Tips

1. **Lazy Loading**: Use Next.js Image component or lazy loading for better performance
2. **Caching**: Images are static files, so they'll be cached by the browser
3. **Preload**: For critical molecules, preload images in the head
4. **Thumbnails**: Consider generating smaller thumbnails for list views

```typescript
// Next.js Image component example
import Image from 'next/image';

const OptimizedMoleculeImage = ({ realId, moleculeType = 'primary' }) => {
  return (
    <Image
      src={`/data/molecule_images/${realId}_${moleculeType}.png`}
      alt={`Molecule ${realId}`}
      width={400}
      height={400}
      loading="lazy"
    />
  );
};
```

