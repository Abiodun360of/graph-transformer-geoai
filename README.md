# 🌍 Graph-Transformer GeoAI for Spatiotemporal Flood Risk Mapping

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.3+-orange?style=flat-square)](https://pyg.org/)
[![GEE](https://img.shields.io/badge/Google_Earth_Engine-API-green?style=flat-square)](https://earthengine.google.com/)
[![Colab](https://img.shields.io/badge/Google_Colab-T4_GPU-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research_Complete-brightgreen?style=flat-square)](https://github.com/Abiodun360of/graph-transformer-geoai)

### A novel spatiotemporal deep learning framework combining Graph Neural Networks and Temporal Transformers for flood risk mapping in sub-Saharan Africa

[📄 Paper](#paper) · [🚀 Run on Colab](#run-on-colab) · [📊 Results](#results) · [🏗️ Architecture](#architecture)

---

## 📌 Overview

> *Most flood risk models treat the landscape as independent pixels. This is scientifically wrong — flood risk propagates through connected space along river networks, elevation gradients, and road corridors.*

**Graph-Transformer GeoAI** solves this by representing the landscape as a **spatial graph** and applying a three-stage deep learning pipeline:

| Stage   | Component                  | What it learns                                |
| ------- | -------------------------- | --------------------------------------------- |
| Phase 2 | **CNN Encoder** (ResNet18) | Spectral & texture features per tile          |
| Phase 3 | **GATv2 Graph Attention**  | Spatial relationships between connected tiles |
| Phase 4 | **Temporal Transformer**   | Seasonal change: dry → wet → flood            |

Applied to the **Niger-Benue Confluence, Lokoja, Kogi State, Nigeria** — one of West Africa's most flood-vulnerable regions.

---

## 🏆 Key Results

| Model                              | Accuracy   | Flood F1   | Macro F1   | AUC    |
| ---------------------------------- | ---------- | ---------- | ---------- | ------ |
| Pixel MLP (baseline)               | 98.9%      | 0.8186     | 0.9066     | 0.9860 |
| CNN features only                  | 99.3%      | 0.8929     | 0.9447     | 0.9916 |
| GATv2 — spatial only (Phase 3)     | 99.82%     | 0.9581     | 0.9786     | 0.9998 |
| **Graph-Transformer GeoAI (Full)** | **99.87%** | **0.9697** | **0.9845** | **0.9998** |

All four configurations are **experimentally obtained** on identical train/val/test splits. The complete ablation demonstrates monotonic improvement as each architectural component is added.

**The spatial graph reasoning (GATv2) contributed the largest single gain (+14.0% F1 over the pixel MLP baseline)** — confirming that flood risk is a network phenomenon requiring graph-based modelling, not pixel-based classification.

---

## 🗺️ Flood Risk Prediction Map

The model **independently reconstructed the Niger and Benue river flood plains** using only satellite imagery and graph topology — without being given explicit river network labels as input.

> 📁 See `outputs/maps/flood_risk_map.png`

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│               GRAPH-TRANSFORMER GeoAI PIPELINE               │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  PHASE 1 — Data (Google Earth Engine)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  ┌──────┐  │
│  │ Sentinel-1  │  │ Sentinel-2  │  │ SRTM DEM │  │ JRC  │  │
│  │ SAR (VV/VH) │  │ (NDVI/MNDWI)│  │ Elevation│  │Labels│  │
│  └──────┬──────┘  └──────┬──────┘  └────┬─────┘  └──┬───┘  │
│         └────────────────┴──────────────┴────────────┘       │
│                          │  19-band GeoTIFF                   │
│                          ▼                                    │
│  PHASE 2 — CNN Feature Extraction                            │
│  ┌──────────────────────────────────────────────┐            │
│  │ Modified ResNet18 (18-channel input)          │            │
│  │ 64×64 tile patches → 256-dim embeddings      │            │
│  │ + normalized spectral features + coordinates  │            │
│  │ Output: Node feature matrix X [24336 × 276]  │            │
│  └─────────────────────┬────────────────────────┘            │
│                         │                                     │
│  PHASE 3 — Graph Attention Network                           │
│  ┌──────────────────────────────────────────────┐            │
│  │ Spatial graph: 24,336 nodes, 196,566 edges   │            │
│  │ Edges: queen adjacency + road connectivity   │            │
│  │ GATv2 (3 layers, 4/4/2 heads, residual)      │            │
│  │ Flood F1 = 0.9581  ✓                         │            │
│  └─────────────────────┬────────────────────────┘            │
│                         │                                     │
│  PHASE 4 — Temporal Transformer                              │
│  ┌──────────────────────────────────────────────┐            │
│  │ 3 time steps: dry / wet / flood period       │            │
│  │ Transformer (4 layers, 4 heads, d_model=64)  │            │
│  │ Fused with GATv2 spatial embeddings          │            │
│  │ Flood F1 = 0.9697  ✓                         │            │
│  └─────────────────────┬────────────────────────┘            │
│                         ▼                                     │
│              Flood Risk Map + Attention Maps                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 🗺️ Study Area

**Lokoja, Kogi State, Nigeria — Niger-Benue Confluence**

- **Coordinates:** 6.4°E–7.1°E, 7.4°N–8.1°N
- **Coverage:** ~6,000 km²
- **Why here:** The 2022 Nigeria floods displaced 1.4 million people. Lokoja sits at West Africa's largest river confluence and floods annually during August–October

```
Graph nodes  :  24,336  (500m × 500m tiles)
Graph edges  :  196,566 (spatial + road connectivity)
Class balance:  96.9% No Risk  |  3.1% Flood Risk (763 nodes)
Label source :  JRC Global Surface Water (1984–2023)
```

---

## 📂 Repository Structure

```
graph-transformer-geoai/
│
├── 📓 notebooks/                          ← Run these on Google Colab
│   ├── phase1_data_acquisition.ipynb      # GEE data pipeline + graph construction
│   ├── phase2_cnn_features.ipynb          # ResNet18 CNN feature extraction
│   ├── phase3_gat_training.ipynb          # GATv2 spatial learning (F1=0.9581)
│   └── phase4_transformer_training.ipynb  # Temporal Transformer (F1=0.9697)
│
├── 📊 outputs/
│   ├── figures/
│   │   ├── training_curves_gat.png        # GATv2 training history
│   │   ├── training_curves_transformer.png
│   │   ├── confusion_matrix.png
│   │   └── attention_map.png              # GAT explainability map
│   ├── maps/
│   │   └── flood_risk_map.png             # Final prediction map
│   └── metrics/
│       ├── phase3_metrics.json            # GATv2 test results
│       └── final_metrics.json             # Full pipeline results
│
├── 📄 docs/
│   └── GeoAI_FloodRisk_Paper.pdf          # IEEE-format research paper
│
├── requirements.txt                       # Python dependencies
├── .gitignore
├── LICENSE
├── RESULTS.md                             # Detailed results & ablation study
└── README.md
```

---

## 🚀 Run on Colab

All notebooks run on **free Google Colab** (T4 GPU). No local GPU needed.

### Prerequisites

1. **Google account** with [Earth Engine access](https://signup.earthengine.google.com/) (free for research/education)
2. **Google Drive** — outputs are saved automatically to `My Drive/GeoAI_Kogi/`

### Step-by-Step

**Step 1** — Open Phase 1 in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abiodun360of/graph-transformer-geoai/blob/main/notebooks/phase1_data_acquisition.ipynb)

> ⚡ Before running: **Runtime → Change runtime type → T4 GPU**

**Step 2** — Run phases in order:

| Notebook                            | Runtime | What happens                                                                                                          |
| ----------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------- |
| `phase1_data_acquisition.ipynb`     | ~15 min | Downloads Sentinel-1/2, DEM, JRC data via GEE. Builds spatial grid and graph edges. Exports 19-band GeoTIFF to Drive. |
| `phase2_cnn_features.ipynb`         | ~20 min | Loads GeoTIFF. Extracts 64×64 patches. Runs ResNet18 encoder. Saves node feature matrix [24336×276].                  |
| `phase3_gat_training.ipynb`         | ~25 min | Trains GATv2 on spatial graph. Saves best model checkpoint. Generates flood risk map.                                 |
| `phase4_transformer_training.ipynb` | ~10 min | Trains Temporal Transformer. Fuses with GATv2 embeddings. Generates final prediction map.                             |

**Step 3** — All outputs are saved to your Google Drive under `GeoAI_Kogi/`

---

## 📦 Data Sources

Everything is **free and open access**:

| Source                   | Dataset                       | Access               | Used For                        |
| ------------------------ | ----------------------------- | -------------------- | ------------------------------- |
| ESA Copernicus           | Sentinel-1 SAR GRD            | Google Earth Engine  | Flood change detection (VV, VH) |
| ESA Copernicus           | Sentinel-2 SR                 | Google Earth Engine  | NDVI, NDWI, MNDWI, EVI          |
| USGS                     | SRTM 30m DEM                  | Google Earth Engine  | Elevation, slope, TWI           |
| EU Joint Research Centre | JRC Global Surface Water v1.4 | Google Earth Engine  | Flood risk labels               |
| OpenStreetMap            | Road & river networks         | osmnx Python library | Graph edge construction         |

---

## 🔬 Technical Highlights

### Why Graph Neural Networks for Flood Risk?

Flood risk propagates through **connected landscape elements** — rivers, elevation channels, roads. A pixel classifier treats every 500m tile as independent. A GNN explicitly models the connections: a node receives information from all its spatial neighbors, learning that a tile adjacent to a flooded river faces elevated risk even before it floods.

### Why GATv2 over GCN?

Standard GCN averages neighbor features equally. **GATv2 learns attention weights** — how much each neighbor should influence the current node's prediction. The attention weights are physically interpretable: high-weight connections along river corridors confirm the model learned hydrological connectivity without being told where rivers are.

### Why a Temporal Transformer?

A single-period snapshot cannot distinguish permanently water-covered areas from seasonally flooded zones. The Transformer learns the **change pattern**: dry-season vegetation → wet-season saturation → flood-period SAR backscatter depression. This temporal trajectory is the real flood risk signal.

### How was class imbalance handled?

96.9% of tiles are No Risk. Standard training would collapse to predicting "No Risk" everywhere (97% accuracy, 0% flood detection). Solutions applied:

- Stratified random splits (minority class in every split)
- 10× minority oversampling in training batches
- Inverse-frequency class weights in CrossEntropyLoss
- Label smoothing (ε = 0.05)

---

## 📄 Paper

**"Graph-Transformer GeoAI for Spatiotemporal Flood Risk Mapping: A Case Study of the Niger-Benue Confluence, Kogi State, Nigeria"**

*Ofobutu Abiodun Emmanuel*  
Department of Remote Sensing and GIS, Federal University of Technology, Akure (FUTA), Nigeria

Available in `docs/GeoAI_FloodRisk_Paper.pdf`

### Cite This Work

```bibtex
@article{ofobutu2025graphtransformer,
  title   = {Graph-Transformer {GeoAI} for Spatiotemporal Flood Risk Mapping:
             A Case Study of the {Niger-Benue} Confluence, {Kogi State, Nigeria}},
  author  = {Ofobutu, Abiodun Emmanuel},
  year    = {2025},
  url     = {https://github.com/Abiodun360of/graph-transformer-geoai},
  note    = {Flood F1 = 0.9697, Accuracy = 99.87\%}
}
```

---

## 🛣️ Future Work

- [ ] Weekly Sentinel-1 time series (true temporal resolution)
- [ ] Multi-class severity prediction (Low / Medium / High risk)
- [ ] Extension to Anambra, Bayelsa, Delta states
- [ ] SMOTE-GNN oversampling for graph data
- [ ] Physics-informed edges from DEM flow accumulation
- [ ] Real-time flood monitoring dashboard

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgements

- ESA / Copernicus for free Sentinel-1/2 data
- Google Earth Engine team for planetary-scale computation
- JRC (Pekel et al., 2016) for Global Surface Water dataset
- PyTorch Geometric team for GNN implementations
- OpenStreetMap contributors

---

**Built for flood-vulnerable communities in West Africa 🇳🇬**

*If this work helped you, please ⭐ star the repository*
