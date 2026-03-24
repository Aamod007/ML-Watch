<div align="center">
  <h1>mlwatch 🔭</h1>
  <p><b>A framework-agnostic ML monitoring SDK tailored for production cloud environments.</b></p>
  <p>
    <a href="https://pypi.org/project/mlwatch/"><img src="https://img.shields.io/pypi/v/mlwatch?color=blue&label=PyPI" alt="PyPI version" /></a>
    <a href="https://github.com/Aamod007/ML-Watch/actions"><img src="https://img.shields.io/github/actions/workflow/status/Aamod007/ML-Watch/test.yml?branch=main" alt="Build Status" /></a>
    <a href="https://python.org"><img src="https://img.shields.io/pypi/pyversions/mlwatch" alt="Python Versions" /></a>
    <a href="https://github.com/Aamod007/ML-Watch/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License" /></a>
  </p>
  <p>
    <a href="#key-features">Key Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#integrations">Integrations</a>
  </p>
</div>

---

`mlwatch` is a production-grade, open-source Python library designed to give you end-to-end observability of your machine learning models in production without heavy operational overhead.

Whether your models are built with **PyTorch**, **TensorFlow/Keras**, or **Scikit-Learn**, `mlwatch` plugs seamlessly into your inference path to monitor data drift, track prediction degradation, and alert on system anomalies. It natively exports metrics to **AWS CloudWatch**, **GCP Cloud Monitoring**, or **Azure Monitor**—all while adding virtually zero latency to your predictions via asynchronous background processing.

## 🔥 Key Features

- **📊 Comprehensive Drift Detection**: Out-of-the-box support for Population Stability Index (PSI), Kolmogorov-Smirnov (KS-Test), and Chi-Square tests to monitor feature and prediction drift.
- **🛡️ Out-of-Distribution (OOD) Protection**: Built-in anomaly detection (Isolation Forests & Z-score) catches invalid input data before it hits your inference engine.
- **⚡ Performance Observability**: Tracks throughput (RPS), inference latency (p95, p99), error rates, and resource utilization automatically.
- **☁️ Cloud-Native Metric Export**: First-class, lazy-loaded exporters for AWS, GCP, and Azure right out of the box. No external agents required.
- **🔗 Framework-Agnostic**: Native hooks and callbacks for **PyTorch**, **TensorFlow**, and **Scikit-Learn**. Zero architectural refactoring necessary.
- **🚨 Configurable Alerting**: Define granular metric thresholds and deliver real-time notifications via Slack, PagerDuty, or custom webhooks.
- **🚀 Ultra-low Overhead**: Asynchronous, fail-open design guarantees that monitoring will never block or crash your critical inference path.

---

## 🛠️ Installation

`mlwatch` requires Python `3.8+`. You can install the base package (capable of local JSON/stdout logging) directly via `pip`:

```bash
pip install mlwatch
```

To enable cloud metric exportation, install the corresponding extras. We suggest only installing the necessary dependencies to keep your Docker images lightweight:

```bash
pip install 'mlwatch[aws]'      # AWS CloudWatch support
pip install 'mlwatch[gcp]'      # GCP Cloud Monitoring support
pip install 'mlwatch[azure]'    # Azure Monitor support
pip install 'mlwatch[all]'      # Install all available cloud exporters
```

---

## 🚀 Quick Start

`mlwatch` integrates into your existing prediction code with minimal changes. Below are examples covering the primary supported ML frameworks.

### Scikit-Learn

Wrap any Scikit-Learn `Pipeline` or `Estimator` using `MonitoredPipeline`:

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mlwatch.frameworks.sklearn import MonitoredPipeline

# Load training data to serve as your statistical baseline
X_train = np.random.randn(1000, 10)

# Instantiate and fit your typical sklearn pipeline
model = RandomForestClassifier().fit(X_train, np.random.randint(0, 2, 1000))

# Wrap the model for monitoring
monitored_model = MonitoredPipeline(
    estimator=model,
    cloud="aws",                  # Export metrics to AWS CloudWatch
    baseline_data=X_train,        # Calibrate drift baseline
    thresholds={"psi": 0.20}      # Alert if PSI exceeds 0.20
)

# Use exactly as before; monitoring happens asynchronously
X_prod = np.random.randn(10, 10)
predictions = monitored_model.predict(X_prod)
```

### PyTorch

`mlwatch` automatically registers forward hooks to inspect your `nn.Module` tensors during inference:

```python
from mlwatch import ModelMonitor
import torch

model = get_my_pytorch_model()
X_baseline = torch.randn(1000, 10)

monitor = ModelMonitor(
    model=model, 
    framework="pytorch", 
    cloud="gcp",
    baseline_data=X_baseline
)

# Inference triggers tensor inspection automatically
output = monitor.predict(torch.randn(1, 10))
```

### TensorFlow / Keras

Inject the `MonitorCallback` directly into Keras’ `predict()` lifecycle:

```python
from mlwatch.frameworks.tensorflow import MonitorCallback
import numpy as np

model = get_my_keras_model()
X_baseline = np.random.randn(1000, 10)

# Initialize the callback
callback = MonitorCallback(
    cloud="azure", 
    baseline=X_baseline
)

# Pass callback to model.predict
predictions = model.predict(
    np.random.randn(1, 10), 
    callbacks=[callback]
)
```

### Context Manager API (Custom Usage)

If you are using a custom framework or wish to manually enforce monitoring scope, use the `monitor_session` context manager:

```python
from mlwatch import monitor_session

X_batch = np.random.randn(32, 10)

with monitor_session(model, framework="custom", cloud="local", baseline_data=X_baseline) as session:
    predictions = custom_predict_routine(X_batch)
    # Allows manual logging of arbitrary metrics
    session.log_metric("custom_confidence_score", 0.94)
```

---

## ⚙️ Configuration

`ModelMonitor` exposes extensive configuration to adapt to high-scale production systems:

```python
monitor = ModelMonitor(
    model=my_model,
    framework="sklearn",
    cloud="aws",
    
    # [Data Setup]
    baseline_data=X_train,            # Array-like used to calculate baseline distributions
    
    # [Alerting & Thresholds]
    alert_webhook="https://hooks.slack.com/...", # Slack, PagerDuty, or HTTP webhook
    thresholds={
        "psi": 0.2,                   # Population Stability Index limit
        "ks_pvalue": 0.05,            # Kolmogorov-Smirnov test p-value
        "latency_p95_ms": 500,        # P95 latency (milliseconds) max limit
        "error_rate": 0.01,           # Global error rate cap
        "ood_rate": 0.05,             # Out-of-Distribution input limit
    },
    
    # [Infrastructure & Performance]
    sample_rate=0.2,                  # Only monitor 20% of traffic (for high-throughput)
    async_export=True,                # Keep to True to prevent blocking prediction
    log_inputs=False,                 # Set False (default) to ensure PII remains secure
    namespace="mlwatch-prod",         # Metric namespace grouped in your Cloud provider
)
```

---

## ☁️ Integrations

### AWS CloudWatch
When `cloud="aws"`, `mlwatch` utilizes `boto3`. Make sure your environment has proper access:
- Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` 
- Or attach an **IAM Role** with `CloudWatch:PutMetricData` permission.

### GCP Cloud Monitoring
When `cloud="gcp"`, `mlwatch` authenticates using default application credentials.
- Ensure the `GOOGLE_APPLICATION_CREDENTIALS` environment variable is pointing to a valid service account JSON key containing the `Monitoring Metric Writer` role.

### Azure Monitor
When `cloud="azure"`, `mlwatch` pushes custom metrics using Azure's Ingestion API.
- Assumes `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, and `AZURE_TENANT_ID` are configured to an app registration with the necessary ingestion privileges.

---

## 🤝 Contributing

We welcome community contributions! `mlwatch` aims to expand support to more frameworks and statistical tests. 

1. **Clone the repository:** `git clone https://github.com/Aamod007/ML-Watch.git`
2. **Install dependencies:** `pip install -e .[dev,all]`
3. **Run tests:** `pytest tests/ -v`

---

## 📜 License

This project is licensed under the [Apache 2.0 License](https://github.com/Aamod007/ML-Watch/blob/main/LICENSE) - see the LICENSE file for details.