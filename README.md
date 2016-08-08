# IGA-dealii [![Build Status](https://travis-ci.org/mathLab/IGA-dealii.svg)](https://travis-ci.org/mathLab/IGA-dealii) [![DOI](https://zenodo.org/badge/23774/mathLab/IGA-dealii.svg)](https://zenodo.org/badge/latestdoi/23774/mathLab/IGA-dealii)

Isogeometric Analysis classes for the deal.II library

This repository contains the code used to generate the results of the article 

"Algorithms, data structures and applications for Isogeometric Analysis with the deal.II library"
Marco Tezzele, Nicola Cavallini, Luca Heltai
SISSA - International School for Advanced Studies

## Installation

These examples require deal.II version 8.3 or later to work properly. 

## Poisson example

	cd poisson
	mkdir build
	cd build
    cmake .. -DDEAL_II_DIR=/path/to/dealii/installation
    make

If everything went well, you should have an executable named `poisson`, which you can use to reproduce the results of the article. The executable takes 0 or 5 arguments, respectively:

	./poisson

or

	./poisson finite_element_name quadrature_name degree first_cycle last_cycle

The accepted arguments are:

 * finite_element_name:
 	* bernstein
 	* lagrange
 	* lobatto
 * quadrature_name:
 	* legendre
 	* lobatto
 * degree: degree of the finite element space
 * first_cycle: initial refinement of the grid
 * last_cycle: final refinement of the grid
