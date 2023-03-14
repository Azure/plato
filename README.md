# Plato Toolkit

## Installing the package

The package can be installed by following these steps:

- Cloning the repository in your local machine
- Create a Python virtual environment or Conda environment
- Install the package as follows:
```bash
pip install .
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Coding Standards

In this repository, we use code linting software and require to run tests
before a PR can be merged.

Each time a PR is opened, CI checks run to ensure that the code complies with
PEP8, is type safe, is formatted according to ``black``'s conventions, and
imports are correctly sorted. In addition, the CI pipeline runs unit tests to
assess that the package build correctly and all features work as expected.

To be proactive and not discover that code cannot be merged only when opening
a PR, we suggest to run the following software in your local computer:

- ``flake8``
- ``mypy``
- ``isort``
- ``black``
- ``pytest``

To launch these programs, simply go into the root of the repository and
launch them.
You can automatically lint your code by setting your IDE of choice.
For example, to set VSCode to automatically lint the file you are editing
with ``flake8`` and ``mypy`` follow [this
tutorial](https://code.visualstudio.com/docs/python/linting).
To run ``black`` in VSCode, you can follow [this
tutorial](https://dev.to/adamlombard/how-to-use-the-black-python-code-formatter-in-vscode-3lo0).
A similar process has to be followed for ``isort``.

### Precommit Hooks

To get a smoother developer experience, please install pre-commit hooks in
your environment:
```bash
pip install pre-commit
pre-commit install
```
Now every time you try to commit, the code is linted and issues fixed
automatically when possible, otherwise the offending lines are shown on the
screen.

### Opening a PR

When opening a PR, we require code reviews before merging. To make code
reviews easier, it is recommended that developers adhere to the following
guidelines:

-	Commits should be self-contained and should change just one thing in the code
-	Commit messages should be clear and descriptive of the commit’s purpose
-	PR can contain more commits, but it is always better to have one PR for one commit
-	PR should have a clear description of why the change is made and why it is made in a particular way

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
