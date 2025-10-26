"""Production Gunicorn settings for AstroClassify.

These defaults favour responsive IO-bound workloads (FastAPI + Uvicorn worker) and
add basic worker recycling to avoid zombie processes under prolonged load. Override
individual values using environment variables (GUNICORN_*).
"""
from __future__ import annotations

import multiprocessing
import os


bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:8000")
worker_class = os.environ.get("GUNICORN_WORKER_CLASS", "uvicorn.workers.UvicornWorker")

default_workers = max(2, multiprocessing.cpu_count() // 2)
workers = int(os.environ.get("GUNICORN_WORKERS", default_workers))
threads = int(os.environ.get("GUNICORN_THREADS", "2"))

timeout = int(os.environ.get("GUNICORN_TIMEOUT", "60"))
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", "5"))

max_requests = int(os.environ.get("GUNICORN_MAX_REQUESTS", "2000"))
max_requests_jitter = int(os.environ.get("GUNICORN_MAX_REQUESTS_JITTER", "200"))

loglevel = os.environ.get("GUNICORN_LOGLEVEL", "info")
accesslog = os.environ.get("GUNICORN_ACCESSLOG", "-")
errorlog = os.environ.get("GUNICORN_ERRORLOG", "-")

# Enable reuse_port when running behind Kubernetes/modern kernels for quicker restarts.
reuse_port = os.environ.get("GUNICORN_REUSE_PORT", "true").lower() in {"1", "true", "yes"}
