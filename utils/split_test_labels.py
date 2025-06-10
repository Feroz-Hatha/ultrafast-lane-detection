import json
import os

# Configurable paths
source_file = "TUSimple/test_set/test_label.json"
val_output = "TUSimple/test_set/val_label.json"
test_output = "TUSimple/test_set/real_test_label.json"
split_ratio = 0.3  # 30% for validation

# Read and split
with open(source_file, 'r') as f:
    lines = f.readlines()

split_index = int(len(lines) * split_ratio)
val_lines = lines[:split_index]
test_lines = lines[split_index:]

# Write outputs
with open(val_output, 'w') as f:
    f.writelines(val_lines)

with open(test_output, 'w') as f:
    f.writelines(test_lines)

print(f"Split {len(lines)} samples into:")
print(f" - Validation: {len(val_lines)} → {val_output}")
print(f" - Test: {len(test_lines)} → {test_output}")
