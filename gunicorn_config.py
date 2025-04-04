import multiprocessing

# Gunicorn configuration for Dash app
bind = "0.0.0.0:10000"  # Match the port in cruise_app.py
workers = multiprocessing.cpu_count() * 2 + 1  # Recommended number of workers
worker_class = "gevent"  # Use gevent for async support
timeout = 120  # Increase timeout for long-running operations
keepalive = 5  # Keep connections alive
max_requests = 1000  # Restart workers after handling this many requests
max_requests_jitter = 50  # Add randomness to max_requests
reload = False  # Disable auto-reload in production
preload_app = True  # Load application code before worker processes are forked
accesslog = "-"  # Log to stdout
errorlog = "-"  # Log to stderr
loglevel = "info" 