# GPU Lab Access Guide

## Server Specs
| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (24 GB VRAM) |
| CUDA | 12.4 |
| Python | 3.11 |
| PyTorch | 2.5.1+cu124 |
| SSH Hostname | lab-ssh.shivamaarya.dev |
| Jupyter Hostname | lab-jupyter.shivamaarya.dev |
| User | labuser (has passwordless sudo) |

## Pre-installed Libraries
- sentence-transformers 5.2.3
- transformers 5.2.0
- accelerate 1.12.0
- openai 2.24.0
- anthropic 0.84.0
- numpy 2.1.2, scipy 1.17.1, scikit-learn 1.8.0
- matplotlib 3.10.8, pandas 3.0.1
- dit 1.5
- hdbscan 0.8.41 (installed by us)
- jupyter / jupyterlab

## Setup (one-time per session)
SSH key is at `~/.ssh/ai-lab-key`. SSH config entry `ai-lab` should already exist.

If not:
```
Host ai-lab
    HostName lab-ssh.shivamaarya.dev
    User labuser
    IdentityFile ~/.ssh/ai-lab-key
    ProxyCommand cloudflared access ssh --hostname %h
```

## Quick Commands
```bash
# Connect
ssh ai-lab

# Verify GPU
ssh ai-lab "python -c \"import torch; print(torch.cuda.get_device_name(0))\""

# Upload file
scp -o "ProxyCommand cloudflared access ssh --hostname lab-ssh.shivamaarya.dev" -i ~/.ssh/ai-lab-key local-file.py labuser@lab-ssh.shivamaarya.dev:~/workspace/

# Download file
scp -o "ProxyCommand cloudflared access ssh --hostname lab-ssh.shivamaarya.dev" -i ~/.ssh/ai-lab-key labuser@lab-ssh.shivamaarya.dev:~/workspace/results.csv ./
```

## Workspace Structure
```
/home/labuser/workspace/paper-a/
├── scripts/
│   ├── generate_synthetic_traces.py  (Herald)
│   ├── pid_analysis.py v2            (Herald)
│   ├── measure_entropy.py            (Chaewon)
│   ├── run_debates.py                (Chaewon — TODO)
│   └── self_consistency.py           (Chaewon — TODO)
├── synthetic_traces/                  (4 validated scenarios)
├── entropy_results/                   (validated outputs + plots)
├── pid_results/                       (PID trajectories + plots)
└── debate_traces/                     (real OLMo sessions — TODO)
```

## Notes
- `/home/labuser/workspace` is persistent across container restarts
- labuser has sudo — `apt-get install` works
- HDBSCAN cosine metric not supported in this sklearn version — use euclidean
- PID needs pooling across sessions for meaningful results (10 rounds × 2 agents is too sparse)
