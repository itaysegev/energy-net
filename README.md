# EnergyNet

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/yourusername/EnergyNet/CI)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Simulation](#running-the-simulation)
  - [Interacting with the Environment](#interacting-with-the-environment)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

**EnergyNet** is a sophisticated simulation environment designed to model and analyze smart grid systems. It integrates various components such as battery storage, production units, consumption units, and Independent System Operators (ISOs) to emulate real-world energy distribution and management scenarios. By leveraging reinforcement learning frameworks like Gymnasium, EnergyNet provides a platform for developing and testing intelligent agents aimed at optimizing energy usage, cost, and efficiency.

## Features

- **Modular Architecture:** Easily extendable components including Batteries, Production Units, Consumption Units, and ISOs.

- **Reinforcement Learning Compatibility:** Fully compatible with Gymnasium, enabling seamless integration with RL algorithms.
- **Comprehensive Logging:** Detailed logging mechanisms for monitoring and debugging.
- **Configurable Dynamics:** Choose between model-based and data-driven dynamics for various grid entities.
- **Extensive Testing Suite:** Robust unit tests to ensure reliability and correctness of the simulation components.

## Architecture

![Architecture Diagram](docs/architecture_diagram.png)

EnergyNet's architecture comprises the following key components:

- **Grid Entities:**
  - **Battery:** Manages energy storage with charging and discharging capabilities.
  - **ProductionUnit:** Simulates energy production based on dynamic or deterministic models.
  - **ConsumptionUnit:** Represents energy consumption patterns, adaptable to various scenarios.

- **Independent System Operators (ISOs):**
  - **HourlyPricingISO:** Implements fixed hourly pricing models.
  - **DynamicPricingISO:** Adapts pricing based on real-time energy dynamics.

- **Power Control System Unit (PCSUnit):** Orchestrates interactions between grid entities and ISOs, making decisions to optimize energy distribution.

- **Environment (`EnergyNetEnv`):** Integrates all components into a cohesive simulation environment compatible with Gymnasium.

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **pip** (Python package installer)



