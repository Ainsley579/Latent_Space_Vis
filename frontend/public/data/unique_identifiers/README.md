# Unique Identifiers Summary

## Files Created

1. **unique_identifiers_combined.json** - Complete mapping of real_id → unique information
2. **unique_identifiers_combined.csv** - Same data in CSV format
3. **frontend_lookup.json** - Optimized lookup tables for frontend use
4. **FRONTEND_USAGE.md** - Complete usage guide

## Statistics

- **380 unique data points** (each with unique real_id)
- **376 unique reactions** (rxn_id)
- **156 unique conformers** (catalyst_key)
- **380 unique combinations** (rxn_id + catalyst_key)
- **ddG range:** -1.14 to 4.51

## Key Insight

Each data point is now uniquely identified by:
- **unique_id**: `{rxn_id}_{catalyst_key}` (e.g., "HF-1_1879897-28-5-conformer-1")
- **display_name**: Human-readable format (e.g., "HF-1 (1)")

This allows you to distinguish all 3,000+ data points even though they only have 4 unique products!

## Quick Usage

```typescript
const lookup = await fetch('/data/unique_identifiers/frontend_lookup.json').then(r => r.json());
const data = lookup.by_real_id[realId];
console.log(data.unique_id, data.display_name, data.ddG);
```

See FRONTEND_USAGE.md for complete examples.
