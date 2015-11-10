// ---------------------------------------------------------------------
// $Id: fe_q.h 30036 2013-07-18 16:55:32Z maier $
//
// Copyright (C) 2000 - 2013 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#ifndef __deal2__iga_handler_h
#define __deal2__iga_handler_h


#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_bernstein.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <fstream>

#include "grid_generator.h"


DEAL_II_NAMESPACE_OPEN

/*
 * @author Marco Tezzele, Luca Heltai, 2013
 */


struct UnivariateBezierExtractor
{
  // Degree of the BSpline
  unsigned int degree;

  // Global indices of degrees of freedom
  std::vector<unsigned int> dof_indices;

  // First non zero B-spline basis of a cell
  int first_nonzero_bspline_basis;

  // Local Bezier extractor operator
  FullMatrix<double> local_b_extractor;

  // Total number of degrees of freedom
  unsigned int n_dofs;

  UnivariateBezierExtractor () :
    degree(0),
    first_nonzero_bspline_basis(0),
    n_dofs(0)
  {
  }

  UnivariateBezierExtractor (
    unsigned int degree, unsigned int n_dofs)
    :
    degree(degree),
    dof_indices(degree + 1),
    first_nonzero_bspline_basis(0),
    local_b_extractor(degree + 1, degree + 1),
    n_dofs(n_dofs)

  {
  }

};


// This function performs a Kronecker product between two full matrices given as input
// and returns the resulting full matrix
FullMatrix<double> kronecker_product(const FullMatrix<double> &a,
                                     const FullMatrix<double> &b);

// Misleading name. This function is needed to reorder the dofs when the kronecker
// product between extractors is performed. Nb is the number of degrees of freedom.
// a and b are the vectors of the local dofs of the UnivariateBezierExtractor
std::vector<unsigned int>
kronecker_product(const std::vector<unsigned int> &a,
                  const std::vector<unsigned int> &b,
                  unsigned int Nb);

// See rref matlab
void reduced_row_echelon_form (FullMatrix<double> &matrix,
                               std::vector<unsigned int> &jb,
                               double toll=1e-6);


template<int dim>
struct BezierExtractor
{
  BezierExtractor() :
    extractors(dim),
    dofs_per_cell(1),
    dof_indices(dofs_per_cell),
    first_nonzero_bspline_basis(dim),
    n_dofs(1)
  {};

  BezierExtractor(std::vector<UnivariateBezierExtractor> &ex):
    extractors(ex),
    dofs_per_cell(1),
    first_nonzero_bspline_basis(dim),
    n_dofs(1)
  {
    AssertDimension(dim, ex.size());
    for (unsigned int i=0; i<dim; ++i)
      {
        dofs_per_cell *= (extractors[i].degree+1);
      }

    local_b_extractor = extractors[0].local_b_extractor;
    dof_indices = extractors[0].dof_indices;
    n_dofs = extractors[0].n_dofs;

    for (unsigned int i=1; i<dim; ++i)
      {
        local_b_extractor = kronecker_product(
                              extractors[i].local_b_extractor,
                              local_b_extractor);
        dof_indices = kronecker_product(extractors[i].dof_indices,
                                        dof_indices, n_dofs);
        n_dofs *= extractors[i].n_dofs;
      }
  };

  std::vector<UnivariateBezierExtractor> extractors;
  unsigned int dofs_per_cell;
  FullMatrix<double> local_b_extractor;
  std::vector<unsigned int> dof_indices;
  std::vector<unsigned int> first_nonzero_bspline_basis;
  unsigned int n_dofs;
};


template <int dim, int spacedim>
class IgaHandler
{
public:

  IgaHandler (const std::vector<std::vector<double> > &knot_vectors,
              const std::vector<std::vector<unsigned int> > &mults,
              const unsigned int degree);

//  IgaHandler (const IgaHandler<dim,spacedim> &iga_handler);

  unsigned int n_cells(const std::vector<std::vector<double> > &p) const;

  unsigned int
  n_dofs(const std::vector<std::vector<double> > &p,
         const std::vector<std::vector<unsigned int> > &rep,
         const unsigned int degree) const;

  // Returns the first non zero basis function with support on cell $el
  // on one direction
  unsigned int get_first_basis(const std::vector<unsigned int> &mult,
                               const unsigned int el,
                               const unsigned int degree) const;

  // Compute the coordinates of a single cell in the parametric space given the
  // cell_id and the vector of subdivisions.
  std::vector<unsigned int>
  compute_cell_coordinates(const unsigned int cell_id,
                           const std::vector<unsigned int> &subdivisions);

  void assemble_global_extractor();

  void output_basis();

  void assemble_interpolation_matrices();

  // Returns the knot vectors as an std::vector<std::vector<double> >
  std::vector<std::vector<double> > get_knot_vectors();

  // Returns the vectors of multiplicity of all knots. That is in each vector
  // there are the multiplicity of the knots of the corresponding knot vector.
  // The output is a std::vector<std::vector<unsigned int> >
  std::vector<std::vector<unsigned int> > get_mults();

  // Returns the degree of the spline as an unsigned int
  unsigned int get_degree();

  // Returns the global exctractor as a SparseMatrix<double>
  SparseMatrix<double> get_global_extractor();

  std::map<typename DoFHandler<dim, spacedim>::active_cell_iterator, BezierExtractor<dim> >
  get_iga_objects();

  std::map<typename DoFHandler<dim, spacedim>::active_cell_iterator,
      BezierExtractor<dim> > iga_objects;

  void assemble_constraints_matrix(void);

  // This function transform a vector from the bspline space into the FE space.
  // This is done using the GlobalExtractor G:
  // dts = G^T * src
  void transform_vector_into_fe_space(Vector<double> &dst,
                                      Vector<double> &src);

  // This function transform a vector from the FE space into the bspline space.
  // This is done using the GlobalExtractor G:
  // dts = (G * G^T)^-1 * G * src
  void transform_vector_into_bspline_space(Vector<double> &dst,
                                           Vector<double> &src);

  // This function transform a cell rhs from the FE space into the bspline space.
  // This is done using the LocalExtractor C:
  // dst = C * src
  void transform_cell_rhs_into_bspline_space(Vector<double> &dst,
                                             Vector<double> &src,
                                             typename DoFHandler<dim>::active_cell_iterator &cell);

  // This function transform a cell matrix from the FE space into the bspline space.
  // This is done using the LocalExtractor C:
  // dst = C * src * C^T
  void transform_cell_matrix_into_bspline_space(FullMatrix<double> &dst,
                                                FullMatrix<double> &src,
                                                typename DoFHandler<dim>::active_cell_iterator &cell);



  void project_boundary_values(const typename FunctionMap<spacedim>::type &boundary_function,
                               const Quadrature<dim-1> & q,
                               ConstraintMatrix &constraints);

  void map_dofs_to_support_points(std::vector<Point<dim> > &support_points);


  void distribute_local_to_global(
    FullMatrix<double> &cell_matrix,
    typename DoFHandler<dim>::active_cell_iterator &cell,
    SparseMatrix<double> &system_matrix,
    ConstraintMatrix &constraints);


  void distribute_local_to_global(
    FullMatrix<double> &cell_matrix,
    Vector<double> &cell_rhs,
    typename DoFHandler<dim>::active_cell_iterator &cell,
    SparseMatrix<double> &system_matrix,
    Vector<double> &system_rhs,
    ConstraintMatrix &constraints);


  void make_sparsity_pattern (DynamicSparsityPattern &dsp);


  mutable Threads::Mutex mutex;



// private:

  unsigned int degree;
  unsigned int n_bspline;

  Triangulation<dim, spacedim> tria;
  FE_Bernstein<dim, spacedim> fe;
  DoFHandler<dim, spacedim> dh;
  FESystem<dim, spacedim> fe_sys;
  DoFHandler<dim, spacedim> dh_sys;
  Vector<double> euler;
  MappingFEField<dim, spacedim> *map_fe;

  std::vector<types::global_dof_index> local_dof_indices;

  std::vector<std::vector<double> > knot_vectors;
  std::vector<std::vector<FullMatrix<double> > > local_b_extractors;
  std::vector<std::vector<unsigned int> > mults;

  Quadrature<dim> local_quad;
  FullMatrix<double> BernsteinInterpolation;
  FullMatrix<double> BsplineInterpolation;
  std::vector<Point<dim> > points;

  SparsityPattern sparsity;
  SparseMatrix<double> GlobalExtractor;

  SparsityPattern square_sparsity;
  SparseMatrix<double> square_C_CT;

  ConstraintMatrix constraints_matrix;

  SparseDirectUMFPACK square_global_extractor;

  void compute_local_b_extractors (
    std::vector<FullMatrix<double> > &local_extractors,
    const std::vector<double> &knot_vector);
};



/*@}*/

DEAL_II_NAMESPACE_CLOSE

#endif
