# FLORIS Wake Modeling & Wind Farm Controls

FLORIS is a controls-focused wind farm simulation software incorporating
steady-state engineering wake models into a performance-focused Python
framework.
The software is in active development and engagement with the development team
is highly encouraged. If you are interested in using FLORIS to conduct studies
of a wind farm or extending FLORIS to include your own wake model, please join
the conversation in [GitHub Discussions](https://github.com/NREL/floris/discussions/)!

```{note}
Support for python version 3.8 was dropped in FLORIS v4.3, and support for python version 3.9 will end in the latter half of 2025.
See {ref}`installation` for details. FLORIS v4.3 also made the move to requiring `numpy` version 2. See the [numpy documentation for details](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html).
```

## WETO software

FLORIS is primarily developed with the support from the U.S. Department of Energy and
is part of the `WETO Software Stack <https://nrel.github.io/WETOStack>`_.
For more information and other integrated modeling software, see:

- [Portfolio Overview](https://nrel.github.io/WETOStack/portfolio_analysis/overview.html)
- [Entry Guide](https://nrel.github.io/WETOStack/_static/entry_guide/index.html)
- [Wind Farm Controls Workshop](https://www.youtube.com/watch?v=f-w6whxIBrA&list=PL6ksUtsZI1dwRXeWFCmJT6cEN1xijsHJz)

## Quick Start

FLORIS is a Python package run on the command line typically by providing
an input file with an initial configuration. It can be installed with
```pip install floris``` (see {ref}`installation`). The typical entry point is
{py:class}`.FlorisModel` which accepts the path to the
input file as an argument. From there, changes can be made to the initial
configuration through the {py:meth}`.FlorisModel.set`
routine, and the simulation is executed with
{py:meth}`.FlorisModel.run`.

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
{py:class}`.FlorisModel` such as
{py:meth}`.FlorisModel.get_turbine_layout`,
{py:meth}`.FlorisModel.get_turbine_powers` and
{py:meth}`.FlorisModel.get_farm_AEP`, and
a visualization package is available in {py:mod}`floris.flow_visualization`.
A collection of examples are included in the [repository](https://github.com/NREL/floris/tree/main/examples)
and described in detail in {ref}`examples`.

## Engaging on GitHub

FLORIS leverages the following GitHub features to coordinate support and development efforts:

- [Discussions](https://github.com/NREL/floris/discussions): Collaborate to develop ideas for new use cases, features, and software designs, and get support for usage questions
- [Issues](https://github.com/NREL/floris/issues): Report potential bugs and well-developed feature requests
- [Projects](https://github.com/orgs/NREL/projects/18/): Include current and future work on a timeline and assign a person to "own" it

Generally, the first entry point for the community will be within one of the
categories in Discussions.
[Ideas](https://github.com/NREL/floris/discussions/categories/ideas) is a great spot to develop the
details for a feature request. [Q&A](https://github.com/NREL/floris/discussions/categories/q-a)
is where to get usage support.
[Show and tell](https://github.com/NREL/floris/discussions/categories/show-and-tell) is a free-form
space to show off the things you are doing with FLORIS.
