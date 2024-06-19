.. _api:

=============
API Reference
=============

.. rubric:: Main functions

Functions for computing determinant and related quantities of matrices.

.. autosummary::
    :toctree: generated
    :caption: Main Functions
    :recursive:
    :template: autosummary/member.rst

    detkit.logdet
    detkit.loggdet
    detkit.logpdet
    detkit.memdet

.. rubric:: Supplementary functions

Functions for level-3 matrix operations and matrix factorizations.

.. autosummary::
    :toctree: generated
    :caption: Supplementary
    :recursive:
    :template: autosummary/member.rst

    detkit.orthogonalize
    detkit.ortho_complement
    detkit.lu_factor
    detkit.ldl_factor
    detkit.cho_factor
    detkit.lu_solve
    detkit.ldl_solve
    detkit.cho_solve
    detkit.solve_triangular
    detkit.matmul

.. rubric:: Datasets

Functions to create sample dataset to be used for test and benchmarking purposes.

.. autosummary::
    :toctree: generated
    :caption: Datasets
    :recursive:
    :template: autosummary/member.rst

    detkit.electrocardiogram
    detkit.covariance_matrix
    detkit.design_matrix

.. rubric:: Utilities

Utility functions for profiling memory and process.
   
.. autosummary::
    :toctree: generated
    :caption: Utility Functions
    :recursive:
    :template: autosummary/member.rst

    detkit.get_config
    detkit.get_instructions_per_task
    detkit.Memory
