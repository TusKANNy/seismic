# Seismic Maintenance Manual
This document provides a comprehensive guide to the software maintenance workflow for the Seismic project. It covers the branching model used for development on GitHub and the steps for updating the Python package published on PyPI.

## Branching Model
For source code management, we use a branching model based on the one described in the article [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/).
The main features of our model are described below:
- **Main Branches**:
  - `master`: Always contains the stable version of the software ready for release.
  - `develop`: The main branch for ongoing development, containing the latest code with new features and updates.
<p align="center">
    <img src="imgs/main_branches.png" />
</p>

- **Feature Branches**:
  - Created from `develop` to work on new features.
  - Once a feature is completed, the branch is merged back into `develop` via a pull request.
<p align="center">
    <img src="imgs/feature_branches.png" />
</p>


### Workflow

1. **Creating a feature branch**:
   To work on a new feature, create a branch from `develop`:
   ```bash
   git checkout -b feature/feature-name develop
   ```
2. **Merging the feature back** into `develop`: 
    After completing the feature, merge it back into develop:
    ```bash
    git checkout develop
    git merge --no-ff feature/feature-name
    git branch -d feature/feature-name
    git push origin develop
    ```
3. **Releasing a new stable version**: 
    Periodically, merge develop into master to create a new stable version:
    ```bash
    git checkout master
    git merge --no-ff develop
    git tag -a vX.Y.Z -m "Release version X.Y.Z"
    git push origin master --tags
    ```


## Python Package Update Procedure
The process of releasing the Seismic Python package on PyPI is automated via GitHub Actions.
When a new version is pushed to the `master` branch, the GitHub Action defined in /.github/workflow/maturin_publish.yml will:
1. Build and upload the source distribution.
2. Build and upload binaries in the form of Python Wheels, according to manylinux tags (https://github.com/pypa/manylinux).  
 
The Github Action also allows you to manually trigger workflows from the GitHub UI or via the GitHub API.

To build and publish the soruce distribution and binaries, we used the github action **maturin-action** which allow to install and run a custom maturin command. In this case, we used the maturin publish command. The first execution of the command will build and publish the source distribution in the form of a compressed archive (.tar.gz) and the wheels for x86_64 platforms for all the available Python implementation available (the argument `-f` supplied to the command, will search for all available implementations). As december 2024 manylinux docker images come with CPython 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 3.13 and PyPy 3.10. This Gihub Action is developed and mantained by PyO3 (https://github.com/PyO3/maturin-action).

All subsequent executions of the publish command will avoid the building of soruce distribution by using the `--no-sdist` argument and are used to build Wheels for aarch64 and i686 platform.
As specified in the maturin Action documentation, to use a manylinux image for aarch64 we need to configure QEMU with a setup QEMU action defined in (https://github.com/docker/setup-qemu-action).

Wheels built with a **manylinux2014** docker image are expected to be compatible with linux distros using glibc 2.17 or later. 
We also provide support for Python installation that depends on **musl** on a Linux distribution, by using the **musllinux_1_2** docker image. Wheels are expected to be compatible with linux distros using musl 1.2 or later.