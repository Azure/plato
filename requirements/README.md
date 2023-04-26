# Dependencies

In this folder we find the dependencies needed for working with this project.

There are multiple dependencies defined. One file ``requirements.txt`` is
used to list the dependencies needed by the toolkit itself. The other
``dev_requirements.txt`` is used to list the developer's dependencies needed
to write code for the toolkit.

Dev dependencies are tracked with ``pip-tools``. At the moment, a workflow
for updating packages is not yet defined. Using ``dependabot`` could be the
easier choice. For now, when we want to update a package we need to run again
``pip-tools`` to get the latest version in the various ``requirements.txt``.
For instructions on how to update a package or all packages follow
``pip-tools``' [official docs](https://github.com/jazzband/pip-tools)
