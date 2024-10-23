# Contributing to `SWMManywhere`

Thank you for considering contributing to `SWMManywhere`.

## Bugs

Please [create a new issue](https://github.com/ImperialCollegeLondon/SWMManywhere/issues/new)
if you may have found a bug.
Please describe the bug and instructions on recreating it (including OS and
Python version). Label the issue with `bug`.

## New behaviour

Our intention with `SWMManywhere` is that a high level of customisation to suit
your needs may be achieved by adding new `graphfcns` or new `metrics`, see
below. Other new behaviour may be tagged with `enhancement`, though please
check [existing issues](https://github.com/ImperialCollegeLondon/SWMManywhere/issues)
to see if something similar already exists.

### Take a graph give a graph: `graphfcns`

All transformations that take place do so on graph functions; you can change
the order in which these are executed and add new ones. If you want a
`graphfcn` that does a new thing, please create an issue to discuss with the
label `graphfcn`. If a single new `graphfcn` is not sufficient to capture the
transformations that you'd like to apply, more may be needed. If this is the
case, please first create an issue labelled with `enhancement` detailing the
thing that you would like to capture, where we will discuss what `graphfcns`
are needed, and use this issue to coordinate.

### Evaluate against real data with: `metrics`

We have provided a large set of metrics against which a synthetic graph's
performance may be evaluated if a real network is provided. If you want to
create a new `metric`, please create an issue to discuss with the label
`metric`.

## Installation for development

To install `SWMManywhere` in development mode, first you will need a virtual
environment. Here we use a `conda` environment which lets us use the version of
python we want to use, but you can use any other tool you are familiar with.
Just make sure you use a version of Python compatible with SWMManywhere.

```bash
conda create --name swmmanywhere python=3.10
conda activate swmmanywhere
```

Once in the environment, you need to clone the `SWMManywhere` GitHub repository
locally and move into the right folder. You will need `git` for that, installed
either following the [official instructions](https://git-scm.com/downloads) or
with `conda install git`, if you use `conda`.

```bash
git clone https://github.com/ImperialCollegeLondon/SWMManywhere.git
cd swmmanywhere
```

We use [`pip-tools`](https://pip-tools.readthedocs.io/en/latest/) to ensure
consistency in the development process, ensuring all people contributing to
`SWMManywhere` use the same versions for all the dependencies, which minimises
the conflicts. To install the development dependencies and then `SWMManywhere`
in development mode, run:

```bash
pip install -e .[dev,doc]
```

## Quality assurance and linting

`SWMManywhere` uses a collection of tools that ensure that a specific code
style and formatting is followed throughout the software. The tools we use for
that are [`ruff`](https://docs.astral.sh/ruff/),
[`markdownlint`](https://github.com/igorshubovych/markdownlint-cli),
[`mypy`](https://github.com/pre-commit/mirrors-mypy),
[`refurb`](https://github.com/dosisod/refurb),
[`codespell`](https://github.com/codespell-project/codespell),
[`pyproject-fmt`](https://github.com/tox-dev/pyproject-fmt).
You do not need to run them manually - unless you want to - but rather they are
run automatically every time you make a commit thanks to `pre-commit`.
If you want to run them manually before committing, you can do so with:

```bash
pre-commit run --all-files
```

`pre-commit` should already have been installed when installing the `dev`
dependencies, if you followed the instructions above, but you need to activate
the hooks that `git` will run when making a commit. To do that just run:

```bash
pre-commit install
```

You can customise the checks that `ruff`, `mypy`, and `refurb` will make with
the settings in `pyproject.toml`. For `markdownlint`, you need to edit the
arguments included in the .`pre-commit-config.yaml` file.

## Testing and coverage

`SWMManywhere` uses `pytests` as testing suite. You can run tests by navigating
to the folder and running:

```bash
pytest # run all tests
pytest tests/test_file.py # run a specific file's tests
```

By default the `tests/tests_prepare_data.py` does not test the actual downloads
themselves (since this relies on external APIs actually working at the time of
testing), however downloads can be enabled when testing:

```bash
pytest tests/tests_prepare_data.py -m downloads
```

You can check the coverage for these tests by running:

```bash
coverage run -m pytest
coverage report
```

And generate a new coverage html site for the documentation with

```bash
coverage html
```

## Changing dependencies

As the development process moves forward, you may find you need to add a new
dependency. Just add it to the relevant section of the `pyproject.toml` file.

Read the
[`pip-tools` documentation](https://pip-tools.readthedocs.io/en/latest/) for
more information on the process.
