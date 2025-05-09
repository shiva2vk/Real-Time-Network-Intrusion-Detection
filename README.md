
#  Real-Time Network Intrusion Detection at the Edge using Mamba + CNN 

##  Abstract

This project implements a hybrid deep learning architecture combining **Mamba (state-space sequence model)** and **Convolutional Neural Networks (CNN)** to build an efficient and accurate **Network Intrusion Detection System (NIDS)**. The model is designed to detect anomalies and various types of network attacks using raw or minimally processed flow/session-level features.

By leveraging Mamba’s linear-time sequence modeling and CNN's powerful local feature extraction, this architecture balances **high detection accuracy** with **low-latency inference**, making it ideal for **real-time deployment on edge devices**.

##  Neural Architecture

```
Input (network flow/session/packet window)
        │
   [Preprocessing & Feature Scaling]
        │
       CNN
     (1D/2D conv layers)
        │
     Mamba SSM
  (Selective Scan Modeling)
        │
 Fully Connected Layer
        │
   Softmax / Sigmoid
        ↓
    Classification (Benign / Anomaly or Multiclass)
```

- **CNN Layer**: Captures local patterns and protocol-level features.
- **Mamba Layer**: Captures long-range dependencies and time-step interactions.
- **FC Layer**: Aggregates sequence embeddings for final classification.

##  Use Cases

-  **Network Intrusion Detection (IDS/NIDS)**
-  **Real-time Monitoring of IoT Networks**
-  **Cloud-based Attack Detection**
-  **Lightweight Threat Detection on Routers/Switches**

##  Real-Time & Edge Readiness

- Mamba is **linear-time** and **memory-efficient**, unlike Transformers (quadratic).
- Ideal for **latency-sensitive environments**, such as:
  - Industrial IoT
  - Mobile security appliances
  - SD-WAN edge gateways
- Entire pipeline can be deployed using **ONNX / TensorRT / TFLite** after model export.

##  Key Advantages

| Feature                     | Benefit                                      |
|----------------------------|----------------------------------------------|
|  **Long-range modeling** | Mamba captures sequence dependencies         |
|  **Low compute cost**    | Efficient inference (better than Transformers) |
|  **Small footprint**     | Suitable for edge deployment                 |
|  **High Accuracy**       | On both binary and multi-class attack detection |
|  **Flexible Input**      | Works on raw flows, windowed packets, or HTTP headers |

##  Limitations

| Limitation                        | Description                                 |
|----------------------------------|---------------------------------------------|
|  Model Ecosystem is New        | Fewer tools/pretrained models than Transformers |
|  Performance Gains Are Subtle | May not significantly outperform CNNs on simple datasets |
|  Learning Curve                | Mamba requires SSM-specific understanding    |

##  Dataset Recommendations

- [CIC-IDS2018 / CIC-DDoS2019](https://www.unb.ca/cic/datasets/)
- [CIC-IoT-2023](http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/)
- [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
- [CSIC2010 HTTP Attack Logs](https://github.com/msudol/Web-Application-Attack-Datasets)

##  Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- t-SNE or UMAP visualizations of latent space
- Per-class performance (e.g., DDoS, SQLi, PortScan, etc.)

##  Future Work

- Add adversarial robustness using PGD/FGSM
- Incorporate lightweight self-attention hybrid
- Export to ONNX + optimize with TensorRT for real-time inference

##  Citation

If you use this work, please cite or acknowledge:
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- Stanford + Together Research, 2023



##  License

MIT License. Feel free to use and adapt.

~Vivek
“Security isn't optional — it's learnable.”
