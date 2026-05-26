#!/bin/bash
source ../gpu-simulator/gpgpu-sim/setup_environment

#../gpu-simulator/bin/release/accel-sim.out -trace ./kernelslist.g -config ../gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3070/gpgpusim.config -config ../gpu-simulator/configs/tested-cfgs/SM86_RTX3070/trace.config
../gpu-simulator/bin/release/accel-sim.out -trace ./kernelslist.g -config ./gpgpusim.config -config ./trace.config
