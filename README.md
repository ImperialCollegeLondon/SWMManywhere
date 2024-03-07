# SWMManywhere
<!-- markdown-link-check-disable -->
[![Test and build](https://github.com/ImperialCollegeLondon/SWMManywhere/actions/workflows/ci.yml/badge.svg)](https://github.com/ImperialCollegeLondon/SWMManywhere/actions/workflows/ci.yml)
<!-- markdown-link-check-enable -->

## High level workflow overview

- Specify a bounding box and project name
- Downloaders will download: building outlines, precipitation data, street network, rivers network, elevation data for the bounding box.
- A list of functions (registered `graphfcn`, that take as inputs a graph, file addresses and/or parameters) will transform the rivers + streets graph into a parameterised sewer network graph.
- Write functions will convert this to a SWMM input file
- Sensivitiy analysis will execute a partial (hopefully the quicker portion) of the workflow, to be called 000's of times to conduct SA with SAlib

## List of functions to be applied

- In a config file you will specify a list of functions up to a certain point in the preprocessing procedure (e.g., no point delineating subcatchments in every iteration of sensitivity analysis if you are only changing parameters to do with hydraulic design), which you will run once.
- In another config file you will specify the location of the outputs of this first run, and the remaining functions required to create and run your SWMM model file
- These seems like the preferred approach for a few reasons:
  - Lets you optimise your workflow depending on which parameters you plan to investigate
  - If your workflow is inherently different (e.g., you do hydraulic design in parallel with topology derivation) then this is easily accommodated

## Keeping this here just for reference for now (from the ICL template)

This is a minimal Python 3.10 application that uses [`pip-tools`] for packaging and dependency management. It also provides [`pre-commit`](https://pre-commit.com/) hooks (for for [ruff](https://pypi.org/project/ruff/) and [`mypy`](https://mypy.readthedocs.io/en/stable/)) and automated tests using [`pytest`](https://pytest.org/) and [GitHub Actions](https://github.com/features/actions). Pre-commit hooks are automatically kept updated with a dedicated GitHub Action, this can be removed and replace with [pre-commit.ci](https://pre-commit.ci) if using an public repo. It was developed by the [Imperial College Research Computing Service](https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/).

[`pip-tools`] is chosen as a lightweight dependency manager that adheres to the [latest standards](https://peps.python.org/pep-0621/) using `pyproject.toml`.

## Usage

To use this repository as a template for your own application:

1. Click the green "Use this template" button above
2. Name and create your repository
3. Clone your new repository and make it your working directory
4. Replace instances of `myproject` with your own application name. Edit:
   - `pyproject.toml` (also change the list of authors here)
   - `tests/test_myproject.py`
   - Rename `myproject` directory
5. Create and activate a Virtual Environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate # with Powershell on Windows: `.venv\Scripts\Activate.ps1`
   ```

6. Install development requirements:

   ```bash
   pip install -r dev-requirements.txt
   ```

7. Install the git hooks:

   ```bash
   pre-commit install
   ```

8. Run the main app:

   ```bash
   python -m myproject
   ```

9. Run the tests:

   ```bash
   pytest
   ```

### Updating Dependencies

To add or remove dependencies:

1. Edit the `dependencies` variables in the `pyproject.toml` file (aim to keep development tools separate from the project requirements).
2. Update the requirements files:
   - `pip-compile` for `requirements.txt` - the project requirements.
   - `pip-compile --extra dev -o dev-requirements.txt` for `dev-requirements.txt` - the development requirements.
3. Sync the files with your installation (install packages):
   - `pip-sync dev-requirements.txt requirements.txt`

To upgrade pinned versions, use the `--upgrade` flag with `pip-compile`.

Versions can be restricted from updating within the `pyproject.toml` using standard python package version specifiers, i.e. `"black<23"` or `"pip-tools!=6.12.2"`

### Customising

All configuration can be customised to your preferences. The key places to make changes
for this are:

- The `pyproject.toml` file, where you can edit:
  - The build system (change from setuptools to other packaging tools like [Hatch](https://hatch.pypa.io/) or [flit](https://flit.pypa.io/)).
  - The python version.
  - The project dependencies. Extra optional dependencies can be added by adding another list under `[project.optional-dependencies]` (i.e. `doc = ["mkdocs"]`).
  - The `mypy` and `pytest` configurations.
- The `.pre-commit-config.yaml` for pre-commit settings.
- The `.github` directory for all the CI configuration.
  - This repo uses `pre-commit.ci` to update pre-commit package versions and automatically merges those PRs with the `auto-merge.yml` workflow.
  - Note that `pre-commit.ci` is an external service and free for open source repos. For private repos uncomment the commented portion of the `pre-commit_autoupdate.yml` workflow.

[`pip-tools`]: https://pip-tools.readthedocs.io/en/latest/

### Publishing

The GitHub workflow includes an action to publish on release.
To run this action, uncomment the commented portion of `publish.yml`, and modify the steps for the desired behaviour (publishing a Docker image, publishing to PyPI, deploying documentation etc.)
