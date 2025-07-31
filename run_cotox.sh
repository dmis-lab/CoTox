#!/bin/bash

SCRIPT_NAMES=("CoTox_iupac_gpt_4o.py")

TEST_NAMES=("test_1")

for script in "${SCRIPT_NAMES[@]}"
do
  for test in "${TEST_NAMES[@]}"
  do
    echo "Running $script with --test_name=$test"
    python "$script" --test_name="$test"
  done
done