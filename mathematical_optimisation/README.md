# IGP/MPLS-TE Optimization with Survivability

This project implements **linear programming models** for combined IGP/MPLS-TE routing, including **survivability constraints** and scalability analysis across both real and synthetic networks.

---

## **Features**

### **Core Optimization Model**

* **Linear Programming formulation** for combined IGP/MPLS-TE routing
* **Survivability constraints** to handle single link failure scenarios
* **Delta parameter (δ)** to control the number of LSPs (Label Switched Paths)
* Support for both **nominal** and **survivable** network conditions

---

## **Testing Capabilities**

### 1. Basic Testing (`test.py`)

The script `test.py` allows you to:

* Load and parse **real network topologies** (Atlanta, GÉANT)
* Run optimization with **different δ values**
* Compare **IGP-only vs. combined routing performance**
* Compute and visualize the **improvement in maximum link utilization**
* Generate **network visualization plots**

**Usage:**

```bash
python test.py
```

When executed, this script will:

1. Load the Atlanta and GÉANT network topologies
2. Compute IGP-only routing utilization
3. Optimize combined IGP/MPLS-TE routing
4. Show the resulting improvement in maximum link utilization
5. Automatically generate and display network visualization plots

---

### 2. Scalability Analysis (`scalability.py`)

The script `scalability.py` performs scalability and sensitivity testing on both real and synthetic networks:

* Tests on networks ranging from **8 to 30+ nodes**
* Compares **different network topologies** (Waxman, BA, ER, WS)
* Performs **delta parameter sensitivity analysis**
* Compares **real vs. synthetic network performance**
* Includes **computational complexity analysis**

**Usage:**

```bash
python scalability.py
```

When executed, this script will run:

* **Phase 0:** Test real networks (Atlanta, GÉANT)
* **Phase 0.1:** Delta analysis on real networks
* **Phase 1:** Delta parameter analysis on synthetic networks
* **Phase 2:** Network size scaling (8–30 nodes)
* **Phase 3:** Topology comparison (Waxman, BA, ER, WS)
* **Phase 4:** Real vs. synthetic network comparison

---

## **Network Models**

### **Real Networks**

* **Atlanta:** 15 nodes, 22 links (from SNDlib)
* **GÉANT:** 23 nodes, 36 links (European research network)

### **Synthetic Networks**

* **Waxman:** Geographic random graphs
* **BA:** Barabási-Albert scale-free model
* **ER:** Erdős-Rényi random graphs
* **WS:** Watts-Strogatz small-world model

---

## **Paper Reference**

This implementation is based on:

> **D. Cherubini, A. Fanni, A. Mereu, A. Frangioni, C. Murgia, M.G. Scutellà, P. Zuddas**
> *"Linear programming models for traffic engineering in 100% survivable networks under combined IS-IS/OSPF and MPLS-TE"*,
> Computers & Operations Research, Volume 38, Issue 12, 2011, Pages 1805–1815.


## **License**

MIT License. See [LICENSE](LICENSE) for details.
