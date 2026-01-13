# 🛡️ FleetSentinel

### The Anti-Swarm Defense System for Autonomous Fleets

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/yourusername/fleetsentinel)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Architecture](https://img.shields.io/badge/architecture-event--driven-orange)](https://kafka.apache.org/)

**FleetSentinel** is a defensive middleware designed for Autonomous Vehicle (AV) operators (Waymo, Cruise, Zoox). It detects, analyzes, and mitigates **"Physical Denial of Service" (PDoS)** attacks where bad actors coordinate ride requests to entrap fleets in dead-end streets or facilitate vandalism.

---

## 🚨 The Problem: Physical DDoS

AV fleet optimization algorithms assume ride requests are made in good faith. Adversaries exploit this by "swarming":
1.  **Coordination:** 50+ actors request rides to a topologically constrained location (cul-de-sac, narrow alley) simultaneously.
2.  **Entrapment:** The fleet dispatcher routes vehicles to the zone, creating artificial gridlock.
3.  **Vandalism:** Once trapped, stationary assets become targets for graffiti, sensor damage, or theft.

## 🛠️ The Solution

FleetSentinel sits between the **Ingress API** and the **Dispatch Engine**. It uses spatiotemporal clustering combined with real-time "Context Oracles" (News/Event APIs) to distinguish between a Taylor Swift concert (Legit High Demand) and a Reddit-coordinated prank (Attack).

### Key Features
* **Swarm Detection Engine:** Redis-backed geospatial density analysis.
* **Contextual Verification:** Checks active density against Ticketmaster, City Permits, and Protest Alerts.
* **Economic Deterrence Layer:** Dynamically requires a **$500 Security Deposit** for users entering "Grey Zone" risk areas.
* **Vehicle Escape Protocol:** Edge logic for trapped assets to engage "Turtle Mode" and audio deterrents.

---

## 📐 The Algorithm: Threat Confidence Scoring

FleetSentinel calculates a **Threat Confidence Score ($C_{threat}$)** for every request cluster.

### 1. Spatiotemporal Density ($\rho$)
We model requests as points in a sliding time window $T$. The density $\rho$ at location $x$ is calculated using a kernel function:

$$\rho(x) = \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$$

*Where $K$ is the kernel (e.g., Gaussian) and $h$ is the bandwidth (search radius).*

### 2. Topology Weighting ($W_{topo}$)
Not all streets are equal. We assign weights based on OpenStreetMap (OSM) data:
* **Arterial Road:** $0.2$ (Hard to block)
* **Residential Street:** $1.0$ (Standard)
* **Dead-End / Single Lane:** $2.5$ (High Trap Risk)

### 3. The Contextual Modifier ($\delta_{context}$)
The "Oracle" services return a modifier based on external truth:
* **Confirmed Event (Concert/Game):** $-5.0$ (Reduces threat drastically)
* **Confirmed Civil Unrest:** $+5.0$ (Max threat)
* **No Data:** $0.0$ (Neutral)

### 4. Final Confidence Equation
The final confidence score $C_{threat}$ (normalized $0 \to 1$) determines the action:

$$C_{threat} = \sigma \left( \alpha \cdot \rho(x) \cdot W_{topo}(x) + \delta_{context} \right)$$

*Where $\sigma$ is the sigmoid activation function and $\alpha$ is a scaling factor.*

---

## 🏗️ System Architecture

The system operates on 4 distinct layers:

### Layer 1: The Sentinel Core (Detection)
* **Tech:** Python 3.11, Redis (Geo).
* **Function:** Ingests MQTT/Kafka stream of ride requests. Performs `GEORADIUS` lookups to calculate $\rho(x)$.

### Layer 2: The Oracle (Enrichment)
* **Tech:** External APIs (Ticketmaster, GDELT Project, Social Sentiment).
* **Function:** If $\rho(x) > Threshold$, queries APIs to find a "Why".

### Layer 3: Economic Gate (Deterrence)
* **Tech:** Stripe / Payment Gateway.
* **Function:**
    * IF $0.4 < C_{threat} < 0.8$: **Trigger Liability Challenge**.
    * User must pre-authorize **$500.00** liability hold to proceed.
    * *Result:* Malicious actors churn; legitimate desperate users convert.

### Layer 4: Edge Defense (The "Turtle")
* **Tech:** C++ / PyTorch (On-Vehicle).
* **Function:** If vehicle is physically surrounded:
    1.  **Lockdown:** Doors/Windows secure.
    2.  **Scream:** PA System plays legal warning.
    3.  **Creep:** Override pedestrian gap constraints to move at 0.5mph (Turtle Mode).

---

## 💻 Installation & Usage

### Prerequisites
* Python 3.11+
* Redis Stack (with RedisSearch and RedisJSON)
* Docker

### 1. Clone the repo
```bash
git clone [https://github.com/yourusername/fleetsentinel.git](https://github.com/yourusername/fleetsentinel.git)
cd fleetsentinel

2. Set up Environment
cp .env.example .env
# Configure your REDIS_URL and STRIPE_API_KEY

3. Run the Sentinel
docker-compose up --build

4. Simulation (Test the Algorithm)
We include a script to simulate a "Dead-End Swarm":
python scripts/simulate_attack.py --users 50 --location "37.7749,-122.4194" --topology "dead-end"

Output:
[SENTINEL] Density Spike Detected: 50 reqs / 30s
[ORACLE] No active events found at location.
[TOPOLOGY] High Risk: Dead-End.
[DECISION] C_threat = 0.92 (CRITICAL) -> BLOCKING ALL REQUESTS.

📂 Project Structure
fleetsentinel/
├── src/
│   ├── core/
│   │   ├── sentinel.py       # Main Density Logic
│   │   └── clustering.py     # DBSCAN / Kernel implementations
│   ├── oracles/
│   │   ├── event_api.py      # Ticketmaster wrappers
│   │   └── civic_data.py     # Protest/News scraping
│   ├── gates/
│   │   └── liability.py      # Payment hold logic
│   └── edge/
│       └── vehicle_defense.cpp # Mock C++ vehicle control overrides
├── scripts/
│   └── simulate_attack.py    # Load testing tool
├── tests/
├── docker-compose.yml
└── README.md

🤝 Contributing
This project is a Proof of Concept (PoC) for improving AV fleet safety. Pull requests regarding better context-verification APIs or improved clustering kernels are welcome.
 * Fork the Project
 * Create your Feature Branch (git checkout -b feature/AmazingFeature)
 * Commit your Changes (git commit -m 'Add some AmazingFeature')
 * Push to the Branch (git push origin feature/AmazingFeature)
 * Open a Pull Request
📄 License
Distributed under the MIT License. See LICENSE for more information.
> Disclaimer: This software is for educational and defensive purposes only. It is designed to protect assets, not to facilitate surveillance.
> 

### Next Step
Enjoy your ride :)


