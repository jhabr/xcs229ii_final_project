#!/bin/bash
(python3 "$PYTHONPATH"/experiments/baseline_runner.py 2>&1) >> "$PYTHONPATH"/logs/baseline.log &