# Notebooks

Run these **in order** on Google Colab. Each notebook saves its outputs to Google Drive under `My Drive/GeoAI_Kogi/` for the next phase to load.

## Execution Order

### Phase 1 — `phase1_data_acquisition.ipynb`
**What it does:**
- Authenticates with Google Earth Engine
- Downloads Sentinel-1 SAR (VV, VH, flood change)
- Downloads Sentinel-2 multispectral (NDVI, NDWI, MNDWI, EVI)
- Downloads SRTM DEM (elevation, slope, TWI)
- Loads JRC Global Surface Water flood labels
- Builds 19-band feature stack
- Exports GeoTIFF to Google Drive (~15 min)
- Downloads road/river networks from OpenStreetMap
- Creates 24,336-node spatial grid (500m × 500m)
- Builds graph edges (queen adjacency + road connectivity)

**Saves to Drive:** `kogi_features_100m.tif`, `kogi_nodes.gpkg`, `kogi_graph_structure.pt`, `kogi_roads.gpkg`

---

### Phase 2 — `phase2_cnn_features.ipynb`
**What it does:**
- Loads the 19-band GeoTIFF from Drive
- Extracts 64×64 pixel patches for each of the 24,336 nodes
- Runs modified ResNet18 CNN (18-channel input)
- Produces 256-dim embedding per node
- Concatenates with normalized raw spectral features + coordinates
- Builds PyTorch Geometric Data object (276-dim node features)
- Creates stratified train/val/test splits

**Saves to Drive:** `kogi_pyg_data.pt`, `kogi_encoder.pth`, `kogi_scaler.pkl`

---

### Phase 3 — `phase3_gat_training.ipynb`
**What it does:**
- Loads PyG graph from Phase 2
- Builds 3-layer GATv2 model (276 → 128 → 64 → 2)
- Handles 96.9%/3.1% class imbalance (oversampling + class weights)
- Trains with AdamW + cosine LR + focal loss
- Best epoch 157, **Flood F1 = 0.922**
- Generates flood risk prediction map
- Visualizes GAT attention weights (explainability)

**Saves to Drive:** `flood_gat_best.pth`, `kogi_gat_embeddings.pt`, `kogi_predictions.gpkg`, `flood_risk_map.png`, `attention_map.png`

---

### Phase 4 — `phase4_transformer_training.ipynb`
**What it does:**
- Loads GATv2 embeddings from Phase 3
- Builds 3-timestep seasonal sequences (dry/wet/flood)
- Trains 4-layer Temporal Transformer
- Fuses temporal features with GATv2 spatial embeddings
- Best epoch 3, **Flood F1 = 0.956**
- Generates final flood risk map
- Ablation study comparison table

**Saves to Drive:** `temporal_transformer_best.pth`, `kogi_final_predictions.gpkg`, `final_flood_map.png`

---

## Important Notes

1. **Always set T4 GPU:** Runtime → Change runtime type → T4 GPU
2. **GEE setup required:** Register at [earthengine.google.com](https://earthengine.google.com) — free for research
3. **Run cells one by one:** Do not "Run All" — watch each cell output before proceeding
4. **Colab disconnects:** Sessions timeout after ~90 min idle. Save outputs to Drive frequently.
5. **Phase 1 GEE export:** After starting the export task, check progress at [code.earthengine.google.com](https://code.earthengine.google.com) → Tasks tab. Wait for it to complete before running Phase 2.
