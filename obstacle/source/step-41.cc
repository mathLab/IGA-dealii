/* ---------------------------------------------------------------------
 * $Id: step-41.cc 30526 2013-08-29 20:06:27Z felix.gruber $
 *
 * Copyright (C) 2011 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Joerg Frohne, Texas A&M University and
 *                        University of Siegen, 2011, 2012
 *          Wolfgang Bangerth, Texas A&M University, 2012
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_bernstein.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <list>

#include "grid_generator.h"
#include "iga_handler.h"


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


namespace Step41
{
  using namespace dealii;

  template <int dim>
  class ObstacleProblem
  {
  public:
    ObstacleProblem (IgaHandler<dim,dim> &iga_handler,
                     ConvergenceTable &convergence_table);
    ~ObstacleProblem ();
    void run (unsigned int cycle);

  private:
    void make_grid ();
    void setup_system();
    void assemble_system ();
    void assemble_mass_matrix_diagonal (SparseMatrix<double> &mass_matrix);
    void update_solution_and_constraints ();
    void solve ();
    void output_results (const unsigned int iteration);
    void compute_error (unsigned int cycle);

    IgaHandler<dim,dim>  &iga_handler;

    unsigned int     degree;
    unsigned int     cg_iter;

    Triangulation<dim>   &triangulation;
    FE_Bernstein<dim>    &fe;
    DoFHandler<dim>      &dof_handler;

    MappingFEField<dim>       *mappingfe;
    IndexSet             active_set;

    ConstraintMatrix            bspline_constraints;

    SparseMatrix<double> bspline_system_matrix;
    SparseMatrix<double> bspline_complete_system_matrix;
    SparseMatrix<double> bspline_mass_matrix;

    Vector<double>       mass_lumping;
    Vector<double>       diagonal_of_mass_matrix;

    Vector<double>       bspline_solution;
    Vector<double>       bspline_mass_lumping;
    Vector<double>       bspline_system_rhs;
    Vector<double>       bspline_complete_system_rhs;
    Vector<double>       bspline_contact_force;
    Vector<double>       bspline_lambda;

    SparsityPattern      sparsity_bspline;

    TrilinosWrappers::PreconditionAMG   precondition;

    Vector<double>                 active_set_vector;

    ConvergenceTable              &convergence_table;
  };



  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
    virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                    const unsigned int  component = 0) const;
  };


  template <int dim>
  double Solution<dim>::value (const Point<dim> &p,
                               const unsigned int /*component*/) const
  {
    double value = p(0)*p(0) + p(1)*p(1) - 0.49;

    if ( value > 0 )
      return std::pow( value, 2);
    else
      return 0;
  }

  template <int dim>
  Tensor<1,dim> Solution<dim>::gradient (const Point<dim> &p,
                                         const unsigned int /*component*/) const
  {
    Tensor<1,dim> return_value;

    double value = p(0)*p(0) + p(1)*p(1) - 0.49;

    if ( value > 0)
      {
        return_value[0] = 4 * p(0) * value;
        return_value[1] = 4 * p(1) * value;
      }
    else
      {
        return_value[0] = 0;
        return_value[1] = 0;
      }
    return return_value;
  }



  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };

  template <int dim>
  double RightHandSide<dim>::value (const Point<dim> &p,
                                    const unsigned int component) const
  {
    Assert (component == 0, ExcNotImplemented());

    if (p.distance(Point<dim>()) > 0.5)
      return -8 * (2*p(0)*p(0) + 2*p(1)*p(1) - 0.49);
    else
      return -8 * 0.49 * (1 - p(0)*p(0) - p(1)*p(1) + 0.49);
  }



  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };

  template <int dim>
  double BoundaryValues<dim>::value (const Point<dim> &p,
                                     const unsigned int component) const
  {
    Assert (component == 0, ExcNotImplemented());
    return std::pow( p(0)*p(0) + p(1)*p(1) - 0.49, 2);
  }


  template <int dim>
  class Obstacle : public Function<dim>
  {
  public:
    Obstacle () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };

  template <int dim>
  double Obstacle<dim>::value (const Point<dim> &p,
                               const unsigned int component) const
  {
    Assert (component == 0, ExcNotImplemented());
    return 0;
  }


  template <int dim>
  ObstacleProblem<dim>::ObstacleProblem (IgaHandler<dim,dim> &iga_handler,
                                         ConvergenceTable &convergence_table)
    :
    iga_handler(iga_handler),
    degree(iga_handler.degree),
    triangulation(iga_handler.tria),
    fe (iga_handler.fe),
    dof_handler (iga_handler.dh),
    mappingfe(iga_handler.map_fe),
    convergence_table(convergence_table)
  {}


  template <int dim>
  ObstacleProblem<dim>::~ObstacleProblem ()
  {
    bspline_system_matrix.clear();
    bspline_complete_system_matrix.clear();
    bspline_mass_matrix.clear();
    if (mappingfe)
      delete mappingfe;
  }


  template <int dim>
  void ObstacleProblem<dim>::make_grid ()
  {
    std::cout << std::endl
              << "Degree: "
              << degree
              << std::endl
              << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: "
              << triangulation.n_cells()
              << std::endl;
  }


  template <int dim>
  void ObstacleProblem<dim>::setup_system ()
  {
    dof_handler.distribute_dofs (fe);

    active_set.set_size (iga_handler.n_bspline);
    active_set_vector.reinit(iga_handler.n_bspline);

    std::cout << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl
              << "Number of degrees of freedom IGA: "
              << iga_handler.n_bspline
              << std::endl
              << std::endl;

    DynamicSparsityPattern bspline_sp(iga_handler.n_bspline);
    iga_handler.make_sparsity_pattern (bspline_sp);

    sparsity_bspline.copy_from(bspline_sp);
    bspline_system_matrix.reinit(sparsity_bspline);
    bspline_complete_system_matrix.reinit(sparsity_bspline);

    bspline_solution.reinit (iga_handler.n_bspline);
    bspline_system_rhs.reinit (iga_handler.n_bspline);
    bspline_complete_system_rhs.reinit (iga_handler.n_bspline);

    bspline_contact_force.reinit (iga_handler.n_bspline);
    bspline_lambda.reinit (iga_handler.n_bspline);

    bspline_mass_matrix.reinit (sparsity_bspline);
    assemble_mass_matrix_diagonal(bspline_mass_matrix);
    diagonal_of_mass_matrix.reinit (iga_handler.n_bspline);

    // Instead of extracting the diagonal of the mass matrix, we perform a
    // mass lumping:
    mass_lumping.reinit(iga_handler.n_bspline);
    for (unsigned int i=0; i<mass_lumping.size(); ++i)
      mass_lumping[i] = 1;
    bspline_mass_matrix.vmult(diagonal_of_mass_matrix, mass_lumping);

    // Boundary values
    QGauss<dim-1>  boundary_quad(fe.degree+2);

    std::map<types::global_dof_index,double> boundary_values;

    typename FunctionMap<dim>::type  dirichlet_boundary;

    BoundaryValues<dim> boundary_funct;
    dirichlet_boundary[0] = &boundary_funct;

    iga_handler.project_boundary_values(dirichlet_boundary,
                                        boundary_quad,
                                        bspline_constraints);
  }



  template <int dim>
  void ObstacleProblem<dim>::assemble_system ()
  {
    std::cout << "   Assembling system..." << std::endl;

    bspline_system_matrix = 0;
    bspline_system_rhs    = 0;

    const QGauss<dim>         quadrature_formula(fe.degree+1);
    const RightHandSide<dim>  right_hand_side;

    FEValues<dim>             fe_values (*mappingfe, fe, quadrature_formula,
                                         update_values   | update_gradients |
                                         update_quadrature_points |
                                         update_JxW_values);

    const unsigned int        dofs_per_cell = fe.dofs_per_cell;
    const unsigned int        n_q_points    = quadrature_formula.size();

    FullMatrix<double>        cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>            cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

    for (; cell!=endc; ++cell)
      {
        fe_values.reinit (cell);
        cell_matrix = 0;
        cell_rhs = 0;

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                cell_matrix(i,j) += (fe_values.shape_grad (i, q_point) *
                                     fe_values.shape_grad (j, q_point) *
                                     fe_values.JxW (q_point));

              cell_rhs(i) += (fe_values.shape_value (i, q_point) *
                              right_hand_side.value (fe_values.quadrature_point (q_point)) *
                              fe_values.JxW (q_point));
            }

        iga_handler.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               cell,
                                               bspline_system_matrix,
                                               bspline_system_rhs,
                                               bspline_constraints);
      }
  }



  template <int dim>
  void
  ObstacleProblem<dim>::
  assemble_mass_matrix_diagonal (SparseMatrix<double> &mass_matrix)
  {
    QIterated<dim> quadrature_formula(QTrapez<1>(),fe.degree);

    FEValues<dim>             fe_values (*mappingfe, fe,
                                         quadrature_formula,
                                         update_values   |
                                         update_JxW_values);

    const unsigned int        dofs_per_cell = fe.dofs_per_cell;
    const unsigned int        n_q_points    = quadrature_formula.size();

    FullMatrix<double>        cell_matrix (dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

    for (; cell!=endc; ++cell)
      {
        fe_values.reinit (cell);
        cell_matrix = 0;

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += (fe_values.shape_value (i, q_point) *
                                   fe_values.shape_value (j, q_point) *
                                   fe_values.JxW (q_point));

        iga_handler.distribute_local_to_global(cell_matrix,
                                               cell,
                                               mass_matrix,
                                               bspline_constraints);
      }
  }



  template <int dim>
  void
  ObstacleProblem<dim>::update_solution_and_constraints ()
  {
    std::cout << "   Updating active set..." << std::endl;

    const double penalty_parameter = 100.0;

    bspline_lambda = 0;
    bspline_complete_system_matrix.residual (bspline_lambda,
                                             bspline_solution,
                                             bspline_complete_system_rhs);

    for (unsigned int i=0; i<bspline_contact_force.size(); ++i)
      bspline_contact_force[i] = bspline_lambda[i]/diagonal_of_mass_matrix[i];
    bspline_contact_force *= -1;

    bspline_constraints.clear();
    active_set.clear();
    active_set_vector.reinit(0);
    active_set_vector.reinit(iga_handler.n_bspline);

    const Obstacle<dim> obstacle;
    std::vector<bool>   dof_touched (iga_handler.n_bspline, false);
    std::vector<unsigned int> local_dof_indices(fe.dofs_per_cell);

    std::vector<Point<dim> > bspline_support_points(iga_handler.n_bspline);

    iga_handler.map_dofs_to_support_points(bspline_support_points);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        local_dof_indices = iga_handler.iga_objects[cell].dof_indices;

        for (unsigned int v=0; v<fe.dofs_per_cell; ++v)
          {
            const unsigned int dof_index = local_dof_indices[v];

            if (dof_touched[dof_index] == false)
              dof_touched[dof_index] = true;
            else
              continue;


            const double obstacle_value = obstacle.value (bspline_support_points[dof_index]);
            const double solution_value = bspline_solution (dof_index);

            if (bspline_lambda(dof_index) +
                penalty_parameter *
                diagonal_of_mass_matrix(dof_index) *
                (solution_value - obstacle_value)
                <
                0)
              {
                active_set.add_index (dof_index);
                active_set_vector[dof_index] = 1;
                bspline_constraints.add_line (dof_index);
                bspline_constraints.set_inhomogeneity (dof_index, obstacle_value);

                bspline_solution (dof_index) = obstacle_value;
                bspline_lambda (dof_index) = 0;
              }
          }
      }

    std::cout << "      Size of active set: " << active_set.n_elements()
              << std::endl;

    std::cout << "   Residual of the non-contact part of the system: "
              << bspline_lambda.l2_norm()
              << std::endl;


    // Boundary values
    QGauss<dim-1>  boundary_quad(fe.degree+2);
    std::map<types::global_dof_index,double> boundary_values;

    typename FunctionMap<dim>::type  dirichlet_boundary;

    BoundaryValues<dim> boundary_funct;
    dirichlet_boundary[0] = &boundary_funct;

    iga_handler.project_boundary_values(dirichlet_boundary,
                                        boundary_quad,
                                        bspline_constraints);
  }



  template <int dim>
  void ObstacleProblem<dim>::solve ()
  {
    std::cout << "   Solving system..." << std::endl;

    ReductionControl                    reduction_control (1000, 1e-12, 1e-5);
    SolverCG<Vector<double>>  solver (reduction_control);

    precondition.initialize (bspline_system_matrix);

    solver.solve (bspline_system_matrix, bspline_solution, bspline_system_rhs, precondition);
    bspline_constraints.distribute (bspline_solution);

    cg_iter = reduction_control.last_step();

    std::cout << "      Error: " << reduction_control.initial_value()
              << " -> " << reduction_control.last_value()
              << " in "
              <<  reduction_control.last_step()
              << " CG iterations."
              << std::endl;
  }



  template <int dim>
  void ObstacleProblem<dim>::output_results (const unsigned int iteration)
  {
    std::cout << "   Writing graphical output..." << std::endl;

    DataOut<dim> data_out;

    data_out.attach_dof_handler (dof_handler);

    Vector<double> bspline_sol_dh(dof_handler.n_dofs());
    Vector<double> bspline_solution_vector(bspline_solution);

    iga_handler.transform_vector_into_fe_space(
      bspline_sol_dh,
      bspline_solution_vector);

    data_out.add_data_vector (bspline_sol_dh, "displacement");

    Vector<double> obstacle_vector(triangulation.n_active_cells());

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    unsigned int i=0;
    const Obstacle<dim> obstacle;
    for (; cell!=endc; ++cell, ++i)
      {
        obstacle_vector[i] = obstacle.value(cell->center());
      }

    data_out.add_data_vector (obstacle_vector, "obstacle");

    Vector<double> active_set_dh(dof_handler.n_dofs());
    iga_handler.transform_vector_into_fe_space(
      active_set_dh,
      active_set_vector);
    data_out.add_data_vector (active_set_dh, "active_set");

    Vector<double> bspline_contact_force_dh(dof_handler.n_dofs());
    Vector<double> bspline_contact_force_vector(bspline_contact_force);

    iga_handler.transform_vector_into_fe_space(
      bspline_contact_force_dh,
      bspline_contact_force_vector);

    data_out.add_data_vector (bspline_contact_force_dh, "lambda");

    data_out.build_patches ();

    std::ofstream output_vtk ((std::string("output_") +
                               Utilities::int_to_string (iteration, 3) +
                               ".vtk").c_str ());
    data_out.write_vtk (output_vtk);
  }


  template <int dim>
  void ObstacleProblem<dim>::compute_error (unsigned int cycle)
  {
    std::cout << "   Computing error..." << std::endl;
    Vector<float> difference_per_cell (triangulation.n_active_cells());

    Vector<double> bspline_sol_dh(dof_handler.n_dofs());
    Vector<double> bspline_solution_vector(bspline_solution);

    iga_handler.transform_vector_into_fe_space(
      bspline_sol_dh,
      bspline_solution_vector);

    VectorTools::integrate_difference (dof_handler, bspline_sol_dh,
                                       Solution<dim>(),
                                       difference_per_cell,
                                       QGauss<dim>(2*fe.degree+1),
                                       VectorTools::L2_norm);

    const double L2_error = difference_per_cell.l2_norm();

    VectorTools::integrate_difference (dof_handler, bspline_sol_dh,
                                       Solution<dim>(),
                                       difference_per_cell,
                                       QGauss<dim>(2*fe.degree+1),
                                       VectorTools::H1_seminorm);

    const double H1_error = difference_per_cell.l2_norm();

    const QTrapez<1>     q_trapez;
    const QIterated<dim> q_iterated (q_trapez, 5);
    VectorTools::integrate_difference (dof_handler, bspline_sol_dh,
                                       Solution<dim>(),
                                       difference_per_cell,
                                       q_iterated,
                                       VectorTools::Linfty_norm);

    const double Linfty_error = difference_per_cell.linfty_norm();

    const unsigned int n_active_cells=triangulation.n_active_cells();
    const unsigned int n_dofs=dof_handler.n_dofs();

    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("bsplines", iga_handler.n_bspline);
    convergence_table.add_value("CG", cg_iter);
    convergence_table.add_value("memory_sys", (unsigned int)bspline_system_matrix.memory_consumption());
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);
    convergence_table.add_value("Linfty", Linfty_error);
  }


  template <int dim>
  void ObstacleProblem<dim>::run (unsigned int cycle)
  {
    make_grid();
    setup_system ();

    IndexSet active_set_old (active_set);

    for (unsigned int iteration=0; iteration<=bspline_solution.size (); ++iteration)
      {
        std::cout << "Newton iteration " << iteration << std::endl;

        assemble_system ();

        if (iteration == 0)
          {
            bspline_complete_system_matrix.copy_from (bspline_system_matrix);
            bspline_complete_system_rhs = bspline_system_rhs;
          }

        solve ();
        update_solution_and_constraints ();
        output_results (iteration);

        if (active_set == active_set_old)
          {
            compute_error (cycle);
            break;
          }

        active_set_old = active_set;

        std::cout << std::endl;
      }
  }
}


int main (int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step41;

      deallog.depth_console (0);

      Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);

      // Let's create the mesh
      std::vector<unsigned int> subdivisions(1);
      unsigned int degree = 0;
      bool p_refinement = false;
      bool h_refinement = true;
      bool k_refinement = false;
      ConvergenceTable  convergence_table;

      unsigned int n_cycle = 6;
      Vector<double> times(n_cycle);

      for (unsigned int cycle=1; cycle<n_cycle; ++cycle)
        {
          if (h_refinement)
            {
              subdivisions[0] = std::pow(2, cycle);
              degree = 3;
            }

          if (p_refinement)
            {
              subdivisions[0] = 100;
              degree = cycle;
            }

          if (k_refinement)
            {
              subdivisions[0] = 100;
              degree = 4;
            }

          std::vector<std::vector<double> >
          knots(1, std::vector<double>(subdivisions[0]+1));

          for (unsigned int i=0; i<subdivisions[0]+1; ++i)
            knots[0][i] = -1+i*2./subdivisions[0];

          std::vector<std::vector<unsigned int> >
          mults(1, std::vector<unsigned int>(subdivisions[0]+1, 1));

          // C^0 continuity
          // {
          //   for (unsigned int i=0; i<subdivisions[0]; ++i)
          //     mults[0][i] = degree;
          // }

          // C^1 continuity
          {
           for (unsigned int i=0; i<subdivisions[0]; ++i)
            mults[0][i] = degree-1;
          }

          // C^2 continuity
          // {
          // for (unsigned int i=0; i<subdivisions[0]; ++i)
          //   mults[0][i] = degree-2;
          // }

          if (k_refinement)
            for (unsigned int i=0; i<subdivisions[0]; ++i)
              mults[0][i] = cycle+1;

          // open knot vectors
          mults[0][0] = degree+1;
          mults[0][subdivisions[0]] = degree+1;

          mults.push_back(mults[0]);
          knots.push_back(knots[0]);

          double t0 = seconds();
          IgaHandler<2,2> iga_hd2(knots, mults, degree);
          double t1 = seconds();

          times[cycle] = t1 - t0;

          std::cout << "Time to assemble the IgaHandler: "
                    << times[cycle] << " sec." << std::endl;

          ObstacleProblem<2> obstacle_problem(iga_hd2, convergence_table);

          obstacle_problem.run (cycle);

        }

      convergence_table.set_precision("L2", 3);
      convergence_table.set_precision("H1", 3);
      convergence_table.set_precision("Linfty", 3);
      convergence_table.set_scientific("L2", true);
      convergence_table.set_scientific("H1", true);
      convergence_table.set_scientific("Linfty", true);
      convergence_table.set_tex_caption("cells", "\\# cells");
      convergence_table.set_tex_caption("dofs", "\\# dofs");
      convergence_table.set_tex_caption("bsplines", "\\# B-splines");
      convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
      convergence_table.set_tex_caption("H1", "@f$H^1@f$-error");
      convergence_table.set_tex_caption("Linfty", "@f$L^\\infty@f$-error");
      convergence_table.set_tex_format("cells", "r");
      convergence_table.set_tex_format("dofs", "r");
      convergence_table.set_tex_format("CG", "r");
      convergence_table.set_tex_format("memory_sys", "r");

      std::cout << std::endl
                << "Error "
                << std::endl;
      convergence_table.write_text(std::cout);

      convergence_table.add_column_to_supercolumn("cycle", "n cells");
      convergence_table.add_column_to_supercolumn("cells", "n cells");
      std::vector<std::string> new_order;
      new_order.push_back("n cells");
      new_order.push_back("dofs");
      new_order.push_back("bsplines");
      new_order.push_back("CG");
      new_order.push_back("memory_sys");
      new_order.push_back("L2");
      new_order.push_back("H1");
      new_order.push_back("Linfty");
      convergence_table.set_column_order (new_order);

      std::cout << std::endl
                << "Convergence rate"
                << std::endl;
      convergence_table.write_text(std::cout);

      std::string error_filename = "now_error_deg3_c0";
      if (p_refinement)
        error_filename += "_p_ref.txt";
      if (h_refinement)
        error_filename += "_h_ref.txt";
      if (k_refinement)
        error_filename += "_k_ref.txt";
      std::ofstream error_table_file(error_filename.c_str());
      convergence_table.write_text(error_table_file);


    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
