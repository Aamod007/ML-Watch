# ML-Watch

`mlwatch` is an open-source, framework-agnostic Python library for monitoring machine learning models in production cloud environments. It tracks data drift, prediction degradation, and inference performance, logging metrics safely to AWS CloudWatch, GCP Cloud Monitoring, or Azure Monitor.

## Features

- **Data Drift Detection**: Detect input feature shifts using KS-Test, PSI, and Chi-Square.
- **Performance Metrics**: Track latency, throughput (RPS), and error rates.
- **OOD Detection**: Catch anomalies before inference with Isolation Forest.
- **Cloud Integrations**: Send metrics natively to AWS, GCP, and Azure.
- **Zero-overhead Integration**: Use decorators or context managers to add monitoring without blocking inference.
- **Supported Frameworks**: PyTorch, TensorFlow/Keras, and Scikit-learn.

## Installation

Install the base library for local monitoring:

```bash
pip install mlwatch
```

Install with specific cloud providers:

```bash
pip install mlwatch[aws]        # AWS CloudWatch Support
pip install mlwatch[gcp]        # GCP Cloud Monitoring Support
pip install mlwatch[azure]      # Azure Monitor Support
pip install mlwatch[all]        # All Cloud Providers Supported
```

## Quick Start

### Core API Integration

```python
from mlwatch import ModelMonitor
import numpy as np

# Load your baseline (training data)
X_train = np.random.randn(1000, 10)

# Initialize the monitor
monitor = ModelMonitor(
    model=my_model,               # Your instantiated PyTorch/TF/Sklearn model
    framework="sklearn",          # "pytorch" | "tensorflow" | "sklearn"
    cloud="local",                # "aws" | "gcp" | "azure" | "local"
    baseline_data=X_train,
    thresholds={"psi": 0.2, "latency_p95_ms": 300}
)

# Use it as a drop-in replacement for your predict method
X_input = np.random.randn(10, 10)
predictions = monitor.predict(X_input)
```

### PyTorch using Hook integration

```python
from mlwatch import ModelMonitor

# mlwatch automatically registers PyTorch forward hooks
monitor = ModelMonitor(model=my_pytorch_model, framework="pytorch", cloud="aws")

# Running prediction normally fires hooks automatically
output = monitor.predict(tensor_input)
```

### TensorFlow / Keras Callback

```python
from mlwatch.frameworks.tensorflow import MonitorCallback

callback = MonitorCallback(cloud="gcp", baseline=X_train)
model.predict(X, callbacks=[callback])
```

### Scikit-Learn Wrapper

```python
from mlwatch.frameworks.sklearn import MonitoredPipeline

monitored_pipeline = MonitoredPipeline(my_pipeline, cloud="azure", baseline=X_train)
predictions = monitored_pipeline.predict(X_test)
```

## Configuration

The `ModelMonitor` has several configurable parameters allowing targeted functionality based on your application's setup:

```python
ModelMonitor(
    model,                        
    framework="pytorch",          
    cloud="aws",                  
    baseline_data=X_train,        
    thresholds={                  
        "psi": 0.2,               
        "ks_pvalue": 0.05,        
        "latency_p95_ms": 500,    
        "error_rate": 0.01,       
        "ood_rate": 0.05,         
    },
    sample_rate=1.0,              # Adjust if throughput is massive
    async_export=True,            # Ensure non-blocking inference
    log_inputs=False,             # Disable PII logging
    namespace="mlwatch",          # Custom metric namespace
)
```

## Alerting

You can configure a webhook to receive real-time notifications on drift or degradation threshold breaches:

```python
monitor = ModelMonitor(
    model=my_model,
    framework="pytorch",
    cloud="aws",
    alert_webhook="https://example.com/alerts/webhook"
)
```

## License

This project is licensed under the Apache 2.0 License.