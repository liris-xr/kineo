#!/bin/bash

python kineo/demo/offline/demo.py --sequence-name stone_quarry --batch-size 16 --target-fps 50 --shared-intrinsics \
./assets/stone_quarry_1.mp4 \
./assets/stone_quarry_2.mp4 \
./assets/stone_quarry_3.mp4 \
./assets/stone_quarry_4.mp4 \
./assets/stone_quarry_5.mp4 \
./assets/stone_quarry_6.mp4