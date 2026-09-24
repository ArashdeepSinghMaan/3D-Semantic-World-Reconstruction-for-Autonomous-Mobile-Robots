# 3D Semantic World Reconstruction for Autonomous Mobile Robots

> Building a persistent, geometrically consistent and semantically meaningful 3D representation of the world using camera and LiDAR observations.

## Overview

Autonomous robots need more than 2D object detections or isolated sensor measurements to reason about their surroundings.

A camera provides rich semantic information but represents the world primarily as a 2D image. LiDAR provides direct 3D geometric measurements, but the resulting point cloud is sparse and contains limited semantic information.

This project investigates how these complementary observations can be transformed into a **persistent 3D semantic world representation**.

The objective is not simply to generate a 3D point cloud. The goal is to construct a representation in which the robot can reason about:

- **What** exists in the environment
- **Where** it exists in 3D
- **What its geometric structure is**
- **How large it is**
- **How different objects are spatially related**
- **How confident the system is about its estimates**
- **How observations from different viewpoints contribute to the same world model**

The resulting representation can serve as a perception layer for downstream robotics applications such as navigation, planning, obstacle reasoning, and eventually language-grounded robot interaction.
[Dataset Used](https://fusionportable.github.io/dataset/fusionportable_v2_data/)
---

# 1. Problem Statement

A mobile robot observes the environment through multiple sensors while moving through the world.

At a particular instant:

```text
                 Real World
                     |
          +----------+----------+
          |                     |
       Camera                 LiDAR
          |                     |
      2D image              3D points
          |                     |
    Semantic information     Geometry
          |                     |
          +----------+----------+
                     |
                     v
             3D World Model
