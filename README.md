## Structure

The project is structured as follows:

- `examples/`: Contains examples where controllers run in a centralized or decentralized and communication free fashion.
- `ia_mppi/`: Contains the algorithm that wraps mppi_torch to allow for multi-agent planning.
- `pyproject.toml` and `poetry.lock`: Configuration files for project dependencies.

## Installation

To install the project, follow these steps:

```sh
# Clone the repository
git clone <repository-url>

# Navigate to the project directory
cd <project-directory>

# Install dependencies
poetry install
```

## System-level installation
Alternatively, you can also install at the system level using pip, even though we advise using the virtual environment:
```bash
pip install .
```

## Usage

Access the virtual environment using
```bash
poetry shell
```

To run the point robot example:

```
cd examples/point_robot
python run_centralized.py
```

## Contributing

Contributions are welcome. Please submit a pull request.

## Cite

This repository is a PyTorch re-implementation of the code in
[Multi-Agent Path Integral Control for Interaction-Aware Motion Planning in Urban Canals](https://arxiv.org/abs/2302.06547) 
This PyTorch implementation is easier to extend to different robots, and offers gpu parallelization. The original code, written in C++, could achieve longer planning horizons in real-time, but is not yet available.
If you are using this software, please cite:
```bash
@INPROCEEDINGS{10161511,
  author={Streichenberg, Lucas and Trevisan, Elia and Chung, Jen Jen and Siegwart, Roland and Alonso-Mora, Javier},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Multi-Agent Path Integral Control for Interaction-Aware Motion Planning in Urban Canals}, 
  year={2023},
  volume={},
  number={},
  pages={1379-1385},
  doi={10.1109/ICRA48891.2023.10161511}}
```