# Olympus4SWE `v0.1.0` Release Visuals & Demos

Each Olympus4SWE release archives a self-contained set of architecture and benchmark visualization pages under `docs/releases/<version>/`. Every HTML file in this directory inlines its styles and screenshots, so you can open it directly in a browser without a local server or network connection.

## Included Pages

| Page | Focus | What It Covers |
| :--- | :--- | :--- |
| [architecture.html](architecture.html) | System Architecture Panorama | End-to-end `v0.1.0` system layers: deterministic SDLC control plane, Meta-Harness drivers (`native` and `pi`), Oracle Board, and Ariadne Cockpit. |
| [multi-agent-workflow.html](multi-agent-workflow.html) | Backend Multi-Agent Workflow | Stage 1 `Align`, Stage 2a `Design`, Stage 2 parallel vertical-slice execution, post-DAG unified `Argus` review/repair loop, and clean-directory `verify_product`. |
| [blacknote-v00-vs-v01.html](blacknote-v00-vs-v01.html) | `v0.0` vs. `v0.1` BlackNote Evolution | Side-by-side comparison of the `BlackNote Dev` full-lifecycle benchmark between the `v0.0` baseline and `v0.1.0`. |
| [blacknote-pure-pi-vs-o4e.html](blacknote-pure-pi-vs-o4e.html) | Pure Pi 1.5.2 vs. Olympus4SWE | Side-by-side comparison of standalone Pure Pi 1.5.2 against Olympus4SWE (`run-4e19c8d93539`) on the `BlackNote Dev` task, including rendered screenshots and delivery checks. |

## Viewing Locally

Open any `.html` file directly from a cloned checkout:

```bash
# macOS
open docs/releases/v0.1.0/architecture.html

# Linux
xdg-open docs/releases/v0.1.0/architecture.html
```
