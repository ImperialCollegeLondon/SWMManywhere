# Contributing to `WSIMOD`

Thank you for considering contributing to `WSIMOD`.

## Bugs

Please create a new [issues](https://github.com/ImperialCollegeLondon/wsi/issues) if you may have found a bug. Please describe the bug and instructions on recreating it (including OS and Python version). It may be helpful to use examples from the [tutorials](https://imperialcollegelondon.github.io/wsi/tutorials/) or [how-to's](https://imperialcollegelondon.github.io/wsi/how-to/) to ensure that data is available.

## Confusion

If you are confused about how a model component works, or why it is producing results that look the way they do, please first check the [documentation](https://imperialcollegelondon.github.io/wsi/) and existing [issues](https://imperialcollegelondon.github.io/wsi/issues). If this does not answer your question, or your question has not yet been raised, then please create a new issue where we can discuss it.

## Creating new functionality

Is there something in the water cycle that you would like to represent that is not included in `WSIMOD`? Whatever it is, you are probably not alone! If there is not one already, please create an [issue](https://imperialcollegelondon.github.io/wsi/issues) where we can discuss it. Do this _before_ you start developing as others may be working on the same thing!

Although the development of new functionality will depend highly on the case, there are a few generalisable points to bear in mind:

- `WSIMOD` is highly object-oriented, thus, we will always try to implement a new component as a subclass of the closest component. We will collaboratively discuss this in the issue.
- Our [documentation](https://imperialcollegelondon.github.io/wsi) relies heavily on use of docstrings, make sure to format it following the [Google Python style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), see the source code of [`Land.__init__`](https://imperialcollegelondon.github.io/wsi/reference-land/#wsimod.nodes.land.Land.__init__) for an example. An admin will compile the documentation, but you can create your own pages to be added by following the directions [below](#create-documentation).
- We are incredibly grateful for contributions that include new [tutorials](https://imperialcollegelondon.github.io/wsi/tutorials/) or [how-to's](https://imperialcollegelondon.github.io/wsi/how-to/), whether for new or existing functionality. Our use of the [mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter) extension enables notebooks to form pages in the documentation, but that can also serve as downloadable examples that people can run.
- Design new tests that instantiate your new functionality and test that it produces a specified response. New tests are stored in the `wsi/tests/` folder.

## Installation for development

To install WSIMOD in development mode, first you will need a virtual environment. Here we use a `conda` environment which let us use the version of python we want to use,
but you can use any other tool you are familiar with. Just make sure you use a version of Python compatible with WSIMOD.

```bash
conda create --name wsimod python=3.10
conda activate wsimod
```

Once in the environment, you need to clone the WSIMOD GitHub repository locally and move into the right folder. You will need `git` for that, installed either following the [official instructions](https://git-scm.com/downloads) or with `conda install git`, if you use `conda`.

```bash
git clone https://github.com/ImperialCollegeLondon/wsi.git
cd wsi
```

We use [`pip-tools`](https://pip-tools.readthedocs.io/en/latest/) to ensure consistency in the development process, ensuring all people contributing to WSIMOD uses the same versions for all the dependencies, which minimiese the conflicts. To install the development dependencies and then WISMO in development mode run:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

You can also install the dependencies required to run the demos and tutorials with:

```bash
pip install -r requirements-demos.txt
```

## Quality assurance and linting

WSIMOD uses a collection of tools that ensure that a specific code style and formatting is follow throughout the software. The tools we used for that are [`ruff`](https://docs.astral.sh/ruff/) and [`markdownlint`](https://github.com/igorshubovych/markdownlint-cli). You do not need to run them manually - unless you want to - but rather they are run automatically every time you make a commit thanks to `pre-commit`.

`pre-commit` should already have been installed when installing the `dev` dependencies, if you followed the instructions above, but you need to activate the hooks that `git` will run when making a commit. To do that just run:

```bash
pre-commit install
```

You can customise the checks that `ruff` will make with the settings in `pyproject.toml`. For `markdownlint`, you need to oedit the arguments included in the .`pre-commit-config.yaml` file.

## Testing and coverage

WSIMOD uses `pytests` as testing suite. You can run tests by navigating to the folder and running:

```bash
pytest # run all tests
pytest tests/test_file.py # run a specific file's tests
```

You can check the coverage for these tests by running:

```bash
coverage run -m pytest
coverage report
```

And generate a new coverage html for the documentation with

```bash
coverage html
```

## Create documentation

If you want to compile new documentation you will need some additional packages, installed with:

```bash
pip install -r requirements-doc.txt
```

From here, you can make changes to the documentation pages in `docs` and view how they appear by navigating to and hosting them locally:

```bash
mkdocs serve
```

If compiling and deploying documentation, you will need to have `git` installed (see above). Then:

```bash
mkdocs gh-deploy
```

## Changing dependencies

Is as the development process moves forward you find you need to add a new dependency, just add it to the relevant section of the `pyproject.toml` file and then run `pip-compile` as required to regenerate the different `requirements.txt` files. Read the [`pip-tools` documentation](https://pip-tools.readthedocs.io/en/latest/) for more information on the process.
