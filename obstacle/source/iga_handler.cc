

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
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_bernstein.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <fstream>
#include <iostream>

#include "grid_generator.h"
#include "iga_handler.h"


DEAL_II_NAMESPACE_OPEN

//using namespace dealii;


/* Return the second elapsed since Epoch (00:00:00 UTC, January 1, 1970)
 */
double seconds()
{
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}


FullMatrix<double> kronecker_product(const FullMatrix<double> &a,
                                     const FullMatrix<double> &b)
{
  FullMatrix<double> axb(a.m()*b.m(), a.n()*b.n());
  for (unsigned int i=0; i<a.m(); ++i)
    for (unsigned int j=0; j<a.n(); ++j)
      axb.add(b, a(i,j), i*b.m(), j*b.n());
  return axb;
}

std::vector<unsigned int> kronecker_product(const std::vector<unsigned int> &a,
                                            const std::vector<unsigned int> &b,
                                            unsigned int Nb)
{
  std::vector<unsigned int> axb(a.size()*b.size());

  unsigned int I = 0;
  for (unsigned int i=0; i<a.size(); ++i)
    for (unsigned int j=0; j<b.size(); ++j, ++I)
      axb[I] = a[i]*Nb+b[j];

  return axb;
}


void reduced_row_echelon_form (FullMatrix<double> &matrix,
                               std::vector<unsigned int> &jb,
                               double toll)
{
  unsigned int lead = 0;
  unsigned int i = 0;

  jb.resize(0);

  for (unsigned int row=0; row<matrix.m(); ++row)
    {

      if (lead >= matrix.n())
        break;

      i = row;

      while (abs(matrix[i][lead]) < toll)
        {
          ++i;
          if (i >= matrix.m())
            {
              i = row;
              ++lead;
              if (lead >= matrix.n())
                break;
            }
        }
      if (lead >= matrix.n())
        break;

      jb.push_back(lead);

      matrix.swap_row(i, row);

      // Divide row
      double factor = matrix[row][lead];
      for (unsigned int j=0; j<matrix.n(); ++j)
        matrix[row][j] /= factor;

      for (unsigned int j=0; j<matrix.m(); ++j)
        {
          if (j != row)
            {
              double factor = matrix[j][lead];
              for (unsigned int k=0; k<matrix.n(); ++k)
                {
                  matrix[j][k] += matrix[row][k] * -1 * factor;
                }

            }
        }
    }
  for (unsigned int i=0; i<matrix.m(); ++i)
    for (unsigned int j=0; j<matrix.n(); ++j)
      if (abs(matrix[i][j])<toll)
        matrix[i][j] = 0;
}



template <int dim, int spacedim>
IgaHandler<dim,spacedim>::IgaHandler (const std::vector<std::vector<double> > &knot_vectors,
                                      const std::vector<std::vector<unsigned int> > &mults,
                                      const unsigned int degree)
  :
  degree(degree),
  fe(degree),
  dh(tria),
  fe_sys(FE_Bernstein<dim,spacedim>(degree), spacedim),
  dh_sys (tria),
  map_fe(NULL),
  local_dof_indices(fe.dofs_per_cell),
  knot_vectors(knot_vectors),
  mults(mults),
  local_quad(QIterated<dim>(QTrapez<1>(), 30))
{
  AssertDimension(dim, knot_vectors.size());
  AssertDimension(dim, mults.size());

  local_b_extractors.resize(dim);

  for (int i=0; i<dim; ++i)
    local_b_extractors[i].resize(knot_vectors[i].size()-1,
                                 IdentityMatrix(degree+1));

  GridGenerator::subdivided_hyper_rectangle(tria, knot_vectors);

  GridOut go;
  char name[50];
  sprintf(name, "mesh_%d_%d.msh", dim, spacedim);
  std::ofstream of(name);
  go.write_msh(tria, of);

  dh.distribute_dofs(fe);
  dh_sys.distribute_dofs (fe_sys);

  euler.reinit(dh_sys.n_dofs());
  const ComponentMask mask(spacedim, true);

  VectorTools::get_position_vector(dh_sys, euler, mask);
  map_fe = new MappingFEField<dim,spacedim>(dh_sys, euler, mask);

  std::vector<std::vector<double> > knots_with_repetition(knot_vectors);
  for (int i=0; i<dim; ++i)
    {
      for (int j=mults[i].size()-1; j >= 0; --j)
        knots_with_repetition[i].insert(
          knots_with_repetition[i].begin()+j,
          mults[i][j]-1,
          knots_with_repetition[i][j]);

      compute_local_b_extractors(local_b_extractors[i],
                                 knots_with_repetition[i]);
    }

  std::vector<unsigned int> cell_coordinates(dim);
  std::vector<unsigned int> subdivisions(dim);
  std::vector<UnivariateBezierExtractor> extractors;

  for (int d=0; d<dim; ++d)
    {
      subdivisions[d] = knot_vectors[d].size()-1;
      extractors.push_back(UnivariateBezierExtractor
                           (degree, knots_with_repetition[d].size()-degree-1));
    }

  std::vector<unsigned int> dpo(dim+1, 1U);
  for (unsigned int i=1; i<dpo.size(); ++i)
    dpo[i]=dpo[i-1]*(degree-1);

  std::vector<unsigned int> col_order(Utilities::fixed_power<dim>(degree+1));
  std::vector<unsigned int> row_order(Utilities::fixed_power<dim>(degree+1));

  const FiniteElementData<dim> fe_data(dpo, 1, degree);
  FETools::hierarchic_to_lexicographic_numbering (fe_data, col_order);

  for (unsigned int i=0; i<row_order.size(); ++i)
    row_order[i] = i;

  unsigned int el = 0;
  for (typename DoFHandler<dim, spacedim>::active_cell_iterator
       cell = dh.begin_active(); cell != dh.end(); ++cell, ++el)
    {

      cell_coordinates = compute_cell_coordinates(cell->index(), subdivisions);

      for (unsigned int d=0; d<dim; ++d)
        {
          extractors[d].local_b_extractor = local_b_extractors[d][cell_coordinates[d]];
          for (unsigned int i=0; i<degree+1; ++i)
            extractors[d].dof_indices[i] =
              get_first_basis(mults[d], cell_coordinates[d], degree) + i;
        }

      BezierExtractor<dim> ex(extractors);

      // Reorder them correctly
      iga_objects[cell] = ex;
      iga_objects[cell].local_b_extractor.fill_permutation(
        ex.local_b_extractor,
        row_order,
        col_order);

    }

  unsigned int subdiv = 1;
  n_bspline = 1;
  for (unsigned int i=0; i<dim; ++i)
    {
      subdiv *= subdivisions[i];
      n_bspline *= knots_with_repetition[i].size() - degree - 1;
    }

  assemble_global_extractor();

  square_global_extractor.initialize(square_C_CT);
}


template <int dim, int spacedim>
unsigned int
IgaHandler<dim,spacedim>::n_cells(const std::vector<std::vector<double> > &p) const
{
  unsigned int n=1;
  for (unsigned int i=0; i<p.size(); ++i)
    n *= (p[i].size()-1);
  return n;
};


template <int dim, int spacedim>
unsigned int
IgaHandler<dim,spacedim>::n_dofs(const std::vector<std::vector<double> > &p,
                                 const std::vector<std::vector<unsigned int> > &rep,
                                 const unsigned int degree) const
{
  unsigned int n=1;
  AssertDimension(p.size(), rep.size());
  std::vector<unsigned int> n_dofs(p.size());

  for (unsigned int i=0; i<p.size(); ++i)
    {
      AssertDimension(p[i].size(), rep[i].size());
      for (unsigned int j=0; j<p[i].size(); ++j)
        n_dofs[i] += rep[i][j];
      n_dofs[i] -= (degree+1);
    }


  for (unsigned int i=0; i<p.size(); ++i)
    n *= n_dofs[i];
  return n;
};

// Return the first non zero basis function with support on cell $el
// on one direction
template <int dim, int spacedim>
unsigned int
IgaHandler<dim,spacedim>::get_first_basis(const std::vector<unsigned int> &mult,
                                          const unsigned int el, const unsigned int degree) const
{
  int n=0;
  for (unsigned int i=0; i<el+1; ++i)
    n += mult[i];
  return n-degree-1;
}


template <int dim, int spacedim>
std::vector<unsigned int>
IgaHandler<dim,spacedim>::compute_cell_coordinates(
  const unsigned int cell_id,
  const std::vector<unsigned int> &subdivisions)
{
  std::vector<unsigned int> cell_coordinates(dim);

  switch (dim)
    {
    case 1:
      cell_coordinates[0] = cell_id % subdivisions[0];
      break;
    case 2:
      cell_coordinates[0] = cell_id % subdivisions[0];
      cell_coordinates[1] = (cell_id / subdivisions[0]) % subdivisions[1];
      break;
    case 3:
      cell_coordinates[0] = cell_id % subdivisions[0];
      cell_coordinates[1] = (cell_id / subdivisions[0]) % subdivisions[1];
      cell_coordinates[2] = cell_id / (subdivisions[0] * subdivisions[1]);
      break;
    }

  return cell_coordinates;
}


template <int dim, int spacedim>
void IgaHandler<dim,spacedim>::assemble_global_extractor()
{
  unsigned int n_d = n_dofs(knot_vectors, mults, degree);
  DynamicSparsityPattern cp(dh.n_dofs(),n_d);

  std::vector<unsigned int> dofs(fe.dofs_per_cell);

  for (typename DoFHandler<dim, spacedim>::active_cell_iterator
       cell = dh.begin_active(); cell!=dh.end(); ++cell)
    for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
      for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
        {
          cell->get_dof_indices(dofs);
          cp.add(dofs[j], iga_objects[cell].dof_indices[i]);
        }

  sparsity.copy_from(cp);

  GlobalExtractor.reinit(sparsity);

  for (typename DoFHandler<dim, spacedim>::active_cell_iterator
       cell = dh.begin_active(); cell!=dh.end(); ++cell)
    {
      for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
        for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
          {
            cell->get_dof_indices(dofs);

            GlobalExtractor.set(dofs[j], iga_objects[cell].dof_indices[i],
                                iga_objects[cell].local_b_extractor(i,j));
          }
    }
  DynamicSparsityPattern scp(n_d,n_d);
  square_sparsity.copy_from(scp);
  square_C_CT.reinit(square_sparsity);
  GlobalExtractor.Tmmult(square_C_CT, GlobalExtractor);
}


template <int dim, int spacedim>
void IgaHandler<dim,spacedim>::output_basis()
{
  DataOut<dim, DoFHandler<dim,spacedim> > dataout;
  dataout.attach_dof_handler(dh);
  unsigned int n_d = n_dofs(knot_vectors, mults, degree);
  std::vector<Vector<double> > basis_i_bspline_bezier(n_d);
  for (unsigned int i=0; i<n_d; ++i)
    {
      Vector<double> basis_i_bspline(n_d);
      basis_i_bspline_bezier[i].reinit(dh.n_dofs());
      basis_i_bspline(i) = 1;
      GlobalExtractor.vmult(basis_i_bspline_bezier[i], basis_i_bspline);
      char name[50];
      sprintf(name, "B_%d", i);
      dataout.add_data_vector(basis_i_bspline_bezier[i], name);
    }
  dataout.build_patches(10);
  char fname[50];
  sprintf(fname, "output_%d_%d.vtu", dim, spacedim);
  std::ofstream of(fname);
  dataout.write_vtu(of);
  dataout.clear();
}


template <int dim, int spacedim>
std::vector<std::vector<double> >
IgaHandler<dim,spacedim>::get_knot_vectors ()
{
  return knot_vectors;
}


template <int dim, int spacedim>
std::vector<std::vector<unsigned int> >
IgaHandler<dim,spacedim>::get_mults ()
{
  return mults;
}


template <int dim, int spacedim>
unsigned int
IgaHandler<dim,spacedim>::get_degree ()
{
  return degree;
}


template <int dim, int spacedim>
SparseMatrix<double>
IgaHandler<dim,spacedim>::get_global_extractor ()
{
  return GlobalExtractor;
}


template <int dim, int spacedim>
std::map<typename DoFHandler<dim, spacedim>::active_cell_iterator, BezierExtractor<dim> >
IgaHandler<dim,spacedim>::get_iga_objects()
{
  return iga_objects;
}


template <int dim, int spacedim>
void
IgaHandler<dim,spacedim>::transform_vector_into_fe_space(
  Vector<double> &dst,
  Vector<double> &src)
{
  AssertDimension(dh.n_dofs(), dst.size());
  AssertDimension(n_bspline, src.size());

  GlobalExtractor.vmult(dst, src);
}


template <int dim, int spacedim>
void
IgaHandler<dim,spacedim>::transform_vector_into_bspline_space(
  Vector<double> &dst,
  Vector<double> &src)
{
  AssertDimension(n_bspline, dst.size());
  AssertDimension(dh.n_dofs(), src.size());

  GlobalExtractor.Tvmult(dst, src);
  square_global_extractor.solve(dst);
}


template <int dim, int spacedim>
void
IgaHandler<dim,spacedim>::transform_cell_rhs_into_bspline_space(
  Vector<double> &dst,
  Vector<double> &src,
  typename DoFHandler<dim>::active_cell_iterator &cell)
{
  AssertDimension(fe.dofs_per_cell, dst.size());
  AssertDimension(fe.dofs_per_cell, src.size());

  iga_objects[cell].local_b_extractor.vmult(dst, src);
}


template <int dim, int spacedim>
void
IgaHandler<dim,spacedim>::transform_cell_matrix_into_bspline_space(
  FullMatrix<double> &dst,
  FullMatrix<double> &src,
  typename DoFHandler<dim>::active_cell_iterator &cell)
{
  AssertDimension(fe.dofs_per_cell, dst.m());
  AssertDimension(fe.dofs_per_cell, dst.n());
  AssertDimension(fe.dofs_per_cell, src.m());
  AssertDimension(fe.dofs_per_cell, src.n());

  FullMatrix<double> cell_bspline_matrix_buffer(fe.dofs_per_cell, fe.dofs_per_cell);

  src.mTmult(
    cell_bspline_matrix_buffer,
    iga_objects[cell].local_b_extractor);

  iga_objects[cell].local_b_extractor.mmult(
    dst,
    cell_bspline_matrix_buffer);
}


template <int dim, int spacedim>
void
IgaHandler<dim,spacedim>::project_boundary_values(
  const typename FunctionMap<spacedim>::type &boundary_function,
  const Quadrature<dim-1> & q,
  ConstraintMatrix &constraints)
{
  // Fake boundary values set to 1
  typename FunctionMap<spacedim>::type  fake_boundary;
  ConstantFunction<spacedim> boundary_funct_fake(1);
  fake_boundary[0] = &boundary_funct_fake;

  std::map<types::global_dof_index,double> fake_boundary_values;

  VectorTools::project_boundary_values (dh,
                                        fake_boundary,
                                        q,
                                        fake_boundary_values);

  Vector<double> fake_boundary_values_vector(dh.n_dofs());

  std::map<types::global_dof_index,double>::iterator fake_it = fake_boundary_values.begin();

  for (; fake_it != fake_boundary_values.end(); ++fake_it)
    fake_boundary_values_vector[fake_it->first] = fake_it->second;

  Vector<double> fake_boundary_values_bspline_vector(n_bspline);

  transform_vector_into_bspline_space(fake_boundary_values_bspline_vector,
                                      fake_boundary_values_vector);


  // Real boundary values
  std::map<types::global_dof_index,double> boundary_values;

  VectorTools::project_boundary_values (dh,
                                        boundary_function,
                                        q,
                                        boundary_values);

  Vector<double> boundary_values_vector(dh.n_dofs());

  std::map<types::global_dof_index,double>::iterator it = boundary_values.begin();

  for (; it != boundary_values.end(); ++it)
    boundary_values_vector[it->first] = it->second;

  Vector<double> boundary_values_bspline_vector(n_bspline);

  transform_vector_into_bspline_space(boundary_values_bspline_vector,
                                      boundary_values_vector);


  // Assembling constraints matrix
  for (unsigned int i=0; i<fake_boundary_values_bspline_vector.size(); ++i)
    if (fake_boundary_values_bspline_vector(i) != 0)
      {
        constraints.add_line (i);
        constraints.set_inhomogeneity (i, boundary_values_bspline_vector(i));
      }

  constraints.close();
}


template <int dim, int spacedim>
void
IgaHandler<dim,spacedim>::map_dofs_to_support_points(
  std::vector<Point<dim> > &support_points)
{
  AssertDimension(n_bspline, support_points.size());

  std::vector<std::vector<unsigned int> > mapp(dim,
                                               std::vector<unsigned int>(dh.n_dofs()));

  if (fe.degree > 2)
    {
      // we need to reorder, in order to find correctly the degrees of
      // freedom of the system dof handler.
      FESystem<dim,spacedim> feqs(FE_Q<dim>(fe.degree), spacedim);
      FE_Q<dim,spacedim> feq(fe.degree);
      DoFHandler<dim,spacedim> dhqs(tria);
      dhqs.distribute_dofs(feqs);
      DoFHandler<dim,spacedim> dhq(tria);
      dhq.distribute_dofs(feq);

      typename DoFHandler<dim,spacedim>::active_cell_iterator
      cellq = dhq.begin_active(),
      cellqs = dhqs.begin_active(),
      endc = dhq.end();

      std::vector<unsigned int> dofs_qs(feqs.dofs_per_cell);
      std::vector<unsigned int> dofs_q(feq.dofs_per_cell);
      for (; cellq !=endc; ++cellq, ++cellqs)
        {
          cellq->get_dof_indices(dofs_q);
          cellqs->get_dof_indices(dofs_qs);
          const std::vector<Point<dim> > &sp_qs = feqs.get_unit_support_points();
          const std::vector<Point<dim> > &sp_q = feq.get_unit_support_points();
          for (unsigned int i=0; i<feqs.dofs_per_cell; ++i)
            for (unsigned int k=0; k<feq.dofs_per_cell; ++k)
              if (sp_qs[i].distance(sp_q[k]) < 1e-6)
                mapp[feqs.system_to_component_index(i).first]
                [dofs_q[k]] = dofs_qs[i];
        }
    }
  else
    {
      for (unsigned int d=0; d<dim; ++d)
        for (unsigned int i=0; i<dh.n_dofs(); ++i)
          mapp[d][i] = i*dim+d;
    }


  // Bspline support points
  std::vector<Vector<double> > bspline_euler(dim,
                                             Vector<double>(n_bspline));

  std::vector<Vector<double> > euler_comp(dim,
                                          Vector<double>(dh.n_dofs()));

  for (unsigned int d=0; d<dim; ++d)
    {
      for (unsigned int i=0; i<dh.n_dofs(); ++i)
        euler_comp[d][i] = euler(mapp[d][i]);

      transform_vector_into_bspline_space(bspline_euler[d],
                                          euler_comp[d]);

      for (unsigned int i=0; i<n_bspline; ++i)
        support_points[i][d] = bspline_euler[d][i];
    }
}


template <int dim, int spacedim>
void
IgaHandler<dim,spacedim>::distribute_local_to_global(
  FullMatrix<double> &cell_matrix,
  typename DoFHandler<dim>::active_cell_iterator &cell,
  SparseMatrix<double> &system_matrix,
  ConstraintMatrix &constraints)
{
  AssertDimension(fe.dofs_per_cell, cell_matrix.m());
  AssertDimension(fe.dofs_per_cell, cell_matrix.n());

  FullMatrix<double> cell_bspline_matrix(fe.dofs_per_cell, fe.dofs_per_cell);

  transform_cell_matrix_into_bspline_space(cell_bspline_matrix, cell_matrix, cell);

  constraints.distribute_local_to_global(cell_bspline_matrix,
                                         iga_objects[cell].dof_indices,
                                         system_matrix);
}


template <int dim, int spacedim>
void
IgaHandler<dim,spacedim>::distribute_local_to_global(
  FullMatrix<double> &cell_matrix,
  Vector<double> &cell_rhs,
  typename DoFHandler<dim>::active_cell_iterator &cell,
  SparseMatrix<double> &system_matrix,
  Vector<double> &system_rhs,
  ConstraintMatrix &constraints)
{
  AssertDimension(fe.dofs_per_cell, cell_matrix.m());
  AssertDimension(fe.dofs_per_cell, cell_matrix.n());
  AssertDimension(fe.dofs_per_cell, cell_rhs.size());

  FullMatrix<double> cell_bspline_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
  Vector<double> bspline_cell_rhs(fe.dofs_per_cell);

  transform_cell_matrix_into_bspline_space(cell_bspline_matrix,
                                           cell_matrix,
                                           cell);

  transform_cell_rhs_into_bspline_space(bspline_cell_rhs,
                                        cell_rhs,
                                        cell);

  constraints.distribute_local_to_global(cell_bspline_matrix,
                                         bspline_cell_rhs,
                                         iga_objects[cell].dof_indices,
                                         system_matrix,
                                         system_rhs,
                                         true);
}



template <int dim, int spacedim>
void
IgaHandler<dim,spacedim>::make_sparsity_pattern (DynamicSparsityPattern &dsp)
{
  AssertDimension(n_bspline, dsp.n_rows());
  AssertDimension(n_bspline, dsp.n_cols());

  for (typename DoFHandler<dim>::active_cell_iterator
       cell = dh.begin_active(); cell!=dh.end(); ++cell)
    for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
      for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
        dsp.add(iga_objects[cell].dof_indices[i],
               iga_objects[cell].dof_indices[j]);
}


template <int dim, int spacedim>
void
IgaHandler<dim,spacedim>::assemble_interpolation_matrices ()
{
  FullMatrix<double> local_interpolation_matrix(local_quad.size(),
                                                fe.dofs_per_cell);

  FEValues<dim, spacedim> fe_v(fe, local_quad,
                               update_values | update_quadrature_points);

  std::vector<unsigned int> subdivisions(dim);
  for (int d=0; d<dim; ++d)
    subdivisions[d] = knot_vectors[d].size()-1;

  std::vector<types::global_dof_index> dofs(fe.dofs_per_cell);
// std::vector<types::global_dof_index> bspline_dofs(fe.dofs_per_cell);
  std::vector<unsigned int> cell_coordinates(dim);

  std::vector<unsigned int> first_nonzero_bspline_basis(3);

  unsigned int el = 0;
  for (typename DoFHandler<dim, spacedim>::active_cell_iterator
       cell = dh.begin_active(); cell != dh.end(); ++cell, ++el)
    {
      cell_coordinates = compute_cell_coordinates(cell->index(), subdivisions);

      local_interpolation_matrix = 0;

      cell->get_dof_indices(dofs);
      std::vector<unsigned int> &bspline_dofs =
        iga_objects[cell].dof_indices;

      fe_v.reinit(cell);

      for (unsigned int q = 0; q < local_quad.size(); ++q)
        {
          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            {
              local_interpolation_matrix(q, i) = fe_v.shape_value(i, q);
            }
          for (unsigned int d=0; d<dim; ++d)
            points[el * local_quad.size() + q][d] =
              fe_v.get_quadrature_points()[q][d];
        }


      // Now compute C^e * local_interpolation_matrix, and put it in the
      // right place in BSplineInterpolation
      for (unsigned int q = 0; q < local_quad.size(); ++q)
        {
          for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
            {
              for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                BsplineInterpolation(el * local_quad.size() + q, bspline_dofs[j]) +=
                  local_interpolation_matrix(q, i) *
                  iga_objects[cell].local_b_extractor(j, i);
            }
        }

      for (unsigned int q = 0; q < local_quad.size(); ++q)
        {
          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            {
              BernsteinInterpolation(el * local_quad.size() + q, dofs[i]) +=
                local_interpolation_matrix(q, i);
            }

        }
    }
}


// This function compute the vector containing the local bezier extracting
// operator for each element.
// The algorithm performs a virtual knot insertion of all the knots needed
// to have an equivalent C^0 b-spline formulation.
// input: an open knot vector named knot_vector;
//        a vector of matrices containing the local bezier extractors.
template <int dim, int spacedim>
void
IgaHandler<dim,spacedim>::compute_local_b_extractors (
  std::vector<FullMatrix<double> > &local_extractors,
  const std::vector<double> &knot_vector)
{
  std::vector<unsigned int> mult;
  Assert(knot_vector.size()>0,
         ExcMessage("Cannot do anything with an empty knot vector."));
  unsigned int n_knots = knot_vector.size();
  std::vector<double> alphas(fe.dofs_per_cell);

  unsigned int a = degree;
  unsigned int b = a + 1;
  unsigned int i = 0;
  unsigned int element = 0;

  std::set<double> elements_edges_set(knot_vector.begin(),
                                      knot_vector.end());
  std::vector<double> elements_edges(elements_edges_set.begin(),
                                     elements_edges_set.end());

  unsigned int n_elements = elements_edges.size() - 1;

  mult.resize(n_elements+1);

  // This works only because we assume the knot_vector open
  mult[0] = degree;

  AssertDimension(n_elements, local_extractors.size());

  while (b < n_knots - 1)
    {
      i = b;

      // Count multiplicity of the knot at location b.
      while (b < n_knots - 1 && knot_vector[b + 1] == knot_vector[b])
        {
          ++b;
        }
      mult[element+1] = b - i + 1;

      if (mult[element+1] < degree)
        {
          // Use (10) in Borden to compute the alphas
          double numer = knot_vector[b] - knot_vector[a];

          for (unsigned int j = degree; j > mult[element+1]; --j)
            alphas[j - mult[element+1] - 1] = numer
                                              / (knot_vector[a + j] - knot_vector[a]);

          unsigned int r = degree - mult[element+1];

          // Update the matrix coefficients for r new knots;
          for (unsigned int j = 1; j <= r; ++j)
            {
              unsigned int save = r - j + 1;
              unsigned int s = mult[element+1] + j;

              for (unsigned int k = degree + 1; k > s; --k)
                {
                  double alpha = alphas[k - s - 1];

                  // This corresponds to (9) in Borden
                  for (unsigned int row = 0; row < degree+1; ++row)
                    local_extractors[element][row][k - 1] = alpha
                                                            * local_extractors[element][row][k - 1]
                                                            + (1 - alpha) * local_extractors[element][row][k - 2];

                }

              if (b < n_knots - 1 && element < n_elements - 1)
                {
                  // Update overlapping coefficients of the next operator
                  for (unsigned int row = 0; row <= j; ++row)
                    local_extractors[element + 1][save + row - 1][save - 1] =
                      local_extractors[element][degree - j + row][degree];
                }
            }
          ++element;

          if (b < n_knots - 1)
            {
              // Update indices for the next operator
              a = b;
              ++b;
            }
        }
      else
        {
          ++element;
          a = b;
          ++b;
        }
    }
}



// explicit instantiations
#include "iga_handler.inst"


DEAL_II_NAMESPACE_CLOSE
