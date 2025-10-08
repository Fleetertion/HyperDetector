🚀 HyperDetector
⸻

📖 Overview

HyperDetector is a hypergraph-based APT detection framework designed for web and cloud-scale systems.
It overcomes the limitations of pairwise graph models by capturing multi-entity relations and long-range dependencies.
	•	🔹 Hypergraph Modeling – Represents each system event as a hyperedge linking multiple entities (processes, files, APIs).
	•	🔹 HGNN Encoder – Learns higher-order semantic relations within temporal slices.
	•	🔹 Block Self-Attention – Efficiently correlates distant activities across services.
	•	🔹 Unsupervised KD-Tree Detection – Identifies deviations from normal host behavior.

⸻

⚙️ Environment Setup

git clone https://github.com/Fleetertion/HyperDetector.git
cd HyperDetector
conda create -n hyperdetector python=3.8
conda activate hyperdetector
pip install -r requirements.txt


📂 Datasets
	•	SCVIC-APT-2021 – Realistic multi-domain APT campaign data
	•	wget – Debian host (UNICORN-SC)
	•	DARPA TC E3 – Provenance traces from enterprise networks