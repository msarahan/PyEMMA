"""Utility functions for calculations with multiple thermodynamic state

This is an early prototype of a structure for a low-level package. We should
probably make a similar design as in msmtools and then make the low-level
package part of pytram or pyfeat. This package can use msmtools in order to
avoid duplicate code.

General tasks to be done by this low-level algorithms:


Counting / Connectivity
-----------------------

   * Count transitions between states by thermodynamic state
   * Connectivity over a set of thermodynamic states

Estimation
----------

   * BAR between pairs of thermodynamic states or along a bridge of
     thermodynamic states
   * dTRAM maximum likelihood estimator (given transition counts on a connected
     subset and bias matrix)
   * TRAM MLE (given transition counts on a connected subset and bias matrix)

Analysis
--------

   * Calculate expectation functions

Utilities
---------

   * logsumexp

"""

__author__ = 'noe'

from api import *
