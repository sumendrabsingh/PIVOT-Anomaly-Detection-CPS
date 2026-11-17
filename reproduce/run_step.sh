#!/bin/bash
# run_step.sh <tag> <cmd...>: run one step, log output and exit code.
tag=$1; shift
start=$(date +%s)
"$@" > logs/$tag.log 2>&1
rc=$?
echo "$(date '+%F %T') $tag rc=$rc $(( $(date +%s) - start ))s" >> logs/STATUS
