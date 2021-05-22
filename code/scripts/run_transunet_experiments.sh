#!/bin/bash
if [ -z ${PYTHONPATH+x} ]
then
  echo "PYTHONPATH not set. Exiting..."; exit 1
  else echo "PYTHONPATH set to $PYTHONPATH."
fi
(python3 "$PYTHONPATH"/experiments/transunet_runner.py 2>&1) >> "$PYTHONPATH"/logs/transunet.log &
