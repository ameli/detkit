.. _Run_Tests:

*************
Running Tests
*************

|codecov-devel|

The package can be tested by running the `test scripts <https://github.com/ameli/detkit/tree/main/tests>`_, which tests all `modules <https://github.com/ameli/detkit/tree/main/detkit>`_. 

===========================
Running Tests with `pytest`
===========================


0. Install this package by either of the methods described in the :ref:`installation instructions <Install_Package>`.

1. Install test dependencies:

   ::

       python -m pip install -r tests/requirements.txt

2. Clone the package source code and install the test dependencies:

   ::

       git clone https://github.com/ameli/detkit.git
       cd detkit

3. Test the package by:

   ::

       cd tests
       pytest

   .. warning::

       Do not run tests in the root directory of the package ``/detkit``. To properly run tests, change current working directory to ``/detkit/tests`` sub-directory.

========================
Running Tests with `tox`
========================

To run a test in a virtual environment, use ``tox`` as follows:

1. Install `tox`:
   
   ::
       
       python -m pip install tox

2. Clone the source code from the repository:
   
   ::
       
       git clone https://github.com/ameli/detkit.git

3. run tests by
   
   ::
       
       cd detkit
       tox
  
.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/detkit
   :target: https://codecov.io/gh/ameli/detkit
.. |build-linux| image:: https://github.com/ameli/detkit/workflows/build-linux/badge.svg
   :target: https://github.com/ameli/detkit/actions?query=workflow%3Abuild-linux 
.. |build-macos| image:: https://github.com/ameli/detkit/workflows/build-macos/badge.svg
   :target: https://github.com/ameli/detkit/actions?query=workflow%3Abuild-macos
.. |build-windows| image:: https://github.com/ameli/detkit/workflows/build-windows/badge.svg
   :target: https://github.com/ameli/detkit/actions?query=workflow%3Abuild-windows
