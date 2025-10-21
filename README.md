# FLORIS Wake Modeling and Wind Farm Controls Software

FLORIS is a controls-focused wind farm simulation software incorporating
steady-state engineering wake models into a performance-focused Python
framework. It has been in active development at NREL since 2013 and the latest
release is [FLORIS v.4.5](https://github.com/NREL/floris/releases/latest).
Online documentation is available at https://nrel.github.io/floris.

The software is in active development and engagement with the development team
is highly encouraged. If you are interested in using FLORIS to conduct studies
of a wind farm or extending FLORIS to include your own wake model, please join
the conversation in [GitHub Discussions](https://github.com/NREL/floris/discussions/)!

## WETO software

FLORIS is primarily developed with the support from the U.S. Department of Energy and
is part of the `WETO Software Stack <https://nrel.github.io/WETOStack>`_.
For more information and other integrated modeling software, see:

- [Portfolio Overview](https://nrel.github.io/WETOStack/portfolio_analysis/overview.html)
- [Entry Guide](https://nrel.github.io/WETOStack/_static/entry_guide/index.html)
- [Wind Farm Controls Workshop](https://www.youtube.com/watch?v=f-w6whxIBrA&list=PL6ksUtsZI1dwRXeWFCmJT6cEN1xijsHJz)

## Installation

**If upgrading from a previous version, it is recommended to install FLORIS v4 into a new virtual environment**.
If you intend to use [pyOptSparse](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/) with FLORIS,
it is recommended to install that package first before installing FLORIS.

FLORIS can be installed by downloading the source code or via the PyPI
package manager with `pip`.

The simplest method is with `pip` by using this command:

```bash
pip install floris
```

Developers and anyone who intends to inspect the source code
can install FLORIS by downloading the git repository
from GitHub with ``git`` and use ``pip`` to locally install it.
It is highly recommended to use a Python virtual environment manager
such as [conda](https://docs.conda.io/en/latest/miniconda.html)
in order to maintain a clean and sandboxed environment. The following
commands in a terminal or shell will download and install FLORIS.

```bash
    # Download the source code from the `main` branch
    git clone -b main https://github.com/NREL/floris.git

    # If using conda, be sure to activate your environment prior to installing
    # conda activate <env name>

    # If using pyOptSpare, install it first
    conda install -c conda-forge pyoptsparse

    # Install FLORIS
    pip install -e floris
```

With both methods, the installation can be verified by opening a Python interpreter
and importing FLORIS:

```python
>>> import floris
>>> help(floris)

Help on package floris:

NAME
    floris - # Copyright 2024 NREL

PACKAGE CONTENTS
    convert_floris_input_v3_to_v4
    convert_turbine_v3_to_v4
    core (package)
    cut_plane
    floris_model
    flow_visualization
    layout_visualization
    logging_manager
    optimization (package)
    parallel_floris_model
    turbine_library (package)
    type_dec
    uncertain_floris_model
    utilities
    version
    wind_data

VERSION
    4.5

FILE
    ~/floris/floris/__init__.py
```

It is important to regularly check for new updates and releases as new
features, improvements, and bug fixes will be issued on an ongoing basis.

## Quick Start

FLORIS is a Python package run on the command line typically by providing
an input file with an initial configuration. It can be installed with
```pip install floris``` (see [installation](https://nrel.github.io/floris/installation.html)).
The typical entry point is
[FlorisModel](https://nrel.github.io/floris/_autosummary/floris.floris_model.html)
which accepts the path to the input file as an argument. From there,
changes can be made to the initial configuration through the
[FlorisModel.set](https://nrel.github.io/floris/_autosummary/floris.floris_model.html#floris.floris_model.FlorisModel.set)
routine, and the simulation is executed with
[FlorisModel.run](https://nrel.github.io/floris/_autosummary/floris.floris_model.html#floris.floris_model.FlorisModel.run).

```python
from floris import FlorisModel
fmodel = FlorisModel("path/to/input.yaml")
fmodel.set(
    wind_directions=[i for i in range(10)],
    wind_speeds=[8.0]*10,
    turbulence_intensities=[0.06]*10
)
fmodel.run()
```

Finally, results can be analyzed via post-processing functions available within
[FlorisModel](https://nrel.github.io/floris/_autosummary/floris.floris_model.html#floris.floris_model.FlorisModel)
such as
- [FlorisModel.get_turbine_layout](https://nrel.github.io/floris/_autosummary/floris.floris_model.html#floris.floris_model.FlorisModel.get_turbine_layout)
- [FlorisModel.get_turbine_powers](https://nrel.github.io/floris/_autosummary/floris.floris_model.html#floris.floris_model.FlorisModel.get_turbine_powers)
- [FlorisModel.get_farm_AEP](https://nrel.github.io/floris/_autosummary/floris.floris_model.html#floris.floris_model.FlorisModel.get_farm_AEP)

and in two visualization packages: [layoutviz](https://nrel.github.io/floris/_autosummary/floris.layout_visualization.html) and [flowviz](https://nrel.github.io/floris/_autosummary/floris.flow_visualization.html).
A collection of examples describing the creation of simulations as well as
analysis and post processing are included in the
[repository](https://github.com/NREL/floris/tree/main/examples). Examples are also listed
in the [online documentation](https://nrel.github.io/floris/examples/001_opening_floris_computing_power.html).

## Engaging on GitHub

FLORIS leverages the following GitHub features to coordinate support and development efforts:

- [Discussions](https://github.com/NREL/floris/discussions): Collaborate to develop ideas for new use cases, features, and software designs, and get support for usage questions
- [Issues](https://github.com/NREL/floris/issues): Report potential bugs and well-developed feature requests
- [Projects](https://github.com/orgs/NREL/projects/96): Include current and future work on a timeline and assign a person to "own" it

Generally, the first entry point for the community will be within one of the
categories in Discussions.
[Ideas](https://github.com/NREL/floris/discussions/categories/ideas) is a great spot to develop the
details for a feature request. [Q&A](https://github.com/NREL/floris/discussions/categories/q-a)
is where to get usage support.
[Show and tell](https://github.com/NREL/floris/discussions/categories/show-and-tell) is a free-form
space to show off the things you are doing with FLORIS.


# License

BSD 3-Clause License

Copyright (c) 2024, Alliance for Sustainable Energy LLC, All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions
and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials provided
with the distribution.

* Neither the name of the copyright holder nor the names of its contributors may be used to
endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
