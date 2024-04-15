#!/bin/bash



# download metrics from the cluster

echo "plot the metrics"
# Plot the performance metrics
PY_SCRIPT=../../scripts/plot_metrics_local.py
${PY_SCRIPT} -i metrics.csv -o ./optimize_units_metrics.png
