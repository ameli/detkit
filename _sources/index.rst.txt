******
detkit
******

|licence| |docs|

A toolkit to compute some common functions involving determinant of matrices used in machine learning.

.. toctree::
    :maxdepth: 1
    :caption: Documentation
    :hidden:
    
    Install <install>
    API Reference <api>
    Tests <tests>

=================
List of Functions
=================

.. autosummary::
    :recursive:
    :nosignatures:
    :template: autosummary/member.rst

    detkit.logdet
    detkit.loggdet
    detkit.logpdet

See :ref:`api` for a list of all functions.

========
Features
========

* Functions are implemented with a novel algorithm described in [1]_.
* The underlying library is implemented in C++ and wrapped in cython.
* An accurate count of computational FLOPs during the execution of functions can be measured.

=======
Install
=======

|pypi| |conda-version| |docker|

* Install with `pip <https://pypi.org/project/special-functions/>`_:

  ::
  
      pip install detkit

* Install with `conda <https://anaconda.org/s-ameli/detkit>`:
  
  ::
  
      conda install -c s-ameli detkit
  
* Pull from `docker image <https://hub.docker.com/repository/docker/sameli/detkit>`_
  
  ::
      
      docker pull sameli/detkit

For a complete install or build instructions, see :ref:`install_package`.


====================
Interactive Tutorial
====================

|binder|

Launch an online interactive tutorial in `Jupyter notebook <https://mybinder.org/v2/gh/ameli/detkit/HEAD?filepath=notebooks%2FSpecial%20Functions.ipynb>`_.

=====
Links
=====

* `Package on Anaconda Cloud <https://anaconda.org/s-ameli/detkit>`_
* `Package on PyPi <https://pypi.org/project/detkit/>`_
* `Source code on Github <https://github.com/ameli/detkit>`_
* `Docker image <https://hub.docker.com/repository/docker/sameli/detkit>`_

.. * `Interactive Jupyter notebook <https://mybinder.org/v2/gh/ameli/detkit/HEAD?filepath=notebooks%2FSpecial%20Functions.ipynb>`_.
.. * `API <https://ameli.github.io/detkit/_modules/modules.html>`_

=================
How to Contribute
=================

We welcome contributions via `Github's pull request <https://github.com/ameli/detkit/pulls>`_. If you do not feel comfortable modifying the code, we also welcome feature request and bug report as `Github issues <https://github.com/ameli/detkit/issues>`_.

================
Related Packages
================

* `glean <https://github.com/ameli/glearn>`_: A high-performance python package for machine learning using Gaussian process.
* `imate <https://github.com/ameli/imate>`_: A high-performance python package for implicit matrix trace estimation.

===========
How to Cite
===========


.. [1] Ameli, S., Shadden, S. C. (2022) A Singular Woodbury and
       Pseudo-Determinant Matrix Identities and Application to Gaussian
       Process Regression (`in preparation`).

.. code-block:: latex

    @unpublished{AMELI-2022,
        author = {Ameli, S. and Shadden, S. C.},
        title  = {A Singular {Woodbury} and Pseudo-Determinant Matrix Identities and Application to {Gaussian} Process Regression},
        year   = {2022},
        note   = {in preparation}
    }

================
Acknowledgements
================

* National Science Foundation #1520825
* American Heart Association #18EIA33900046


.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/detkit
   :target: https://codecov.io/gh/ameli/detkit
.. |docs| image:: https://github.com/ameli/detkit/workflows/docs/badge.svg
   :target: https://ameli.github.io/detkit/index.html
.. |licence| image:: https://img.shields.io/github/license/ameli/detkit
   :target: https://opensource.org/licenses/MIT
.. |implementation| image:: https://img.shields.io/pypi/implementation/detkit
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/detkit
.. |format| image:: https://img.shields.io/pypi/format/detkit
.. |pypi| image:: https://img.shields.io/pypi/v/detkit
   :target: https://pypi.org/project/special-functions/
.. |conda| image:: https://anaconda.org/s-ameli/detkit/badges/installer/conda.svg
   :target: https://anaconda.org/s-ameli/detkit
.. |platforms| image:: https://img.shields.io/conda/pn/s-ameli/detkit?color=orange?label=platforms
   :target: https://anaconda.org/s-ameli/detkit
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/detkit
   :target: https://anaconda.org/s-ameli/detkit
.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/ameli/detkit/HEAD?filepath=notebooks%2FSpecial%20Functions.ipynb
.. |downloads| image:: https://pepy.tech/badge/special-functions
   :target: https://pepy.tech/project/detkit
.. |docker| image:: https://img.shields.io/docker/pulls/sameli/detkit
   :target: https://img.shields.io/docker/pulls/sameli/detkit
