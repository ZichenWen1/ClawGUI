#!/bin/bash
# Remove debug print statements added for debugging

echo "Removing debug prints from Sokoban environment..."

# Remove prints from envs.py
sed -i '/print(f"\[WORKER-/d' envs.py
sed -i '/print(f"\[MULTIPROC-/d' envs.py

# Remove prints from projection.py
sed -i '/print(f"\[PROJECTION-/d' projection.py

# Remove prints from sokoban/env.py
sed -i '/print(f"\[ENV-/d' sokoban/env.py

echo "Debug prints removed successfully!"
