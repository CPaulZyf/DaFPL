# Code for "When Federated Prompt Learning Fails: Understanding and Mitigating Cross-Domain Prompt Interference"

This repository provides the implementation of the method proposed in the paper  
**"When Federated Prompt Learning Fails: Understanding and Mitigating Cross-Domain Prompt Interference"**.  

---

## 1. Environment and Dependencies

All experiments have been conducted on the following environment with NVIDIA L40 GPUs:

- Python 3.8.20  
- PyTorch 2.4.1  
- Ubuntu 22.04  


---



## 2. Running Federated Training

To start a  process,  run the following:

```bash
python federated_main.py \
    --dataset "Office" \
    --lambda_dom 0.1 \
    --lambda_unify 0.01 \
    --partition "dir"
