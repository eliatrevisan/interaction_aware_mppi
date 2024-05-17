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

## Usage

To run the point robot example:

```
cd examples/point_robot
python run_centralized.py
```

## Contributing

Contributions are welcome. Please submit a pull request.