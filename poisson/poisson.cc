/* ---------------------------------------------------------------------
 * Copyright (C) 1999 - 2015 by the deal.II authors
 * Copyright (C) 2015 by Marco Tezzele, Nicola Cavallini, Luca Heltai
 *
 * This file has been modified from the example program step-4 of the
 * deal.II library.
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
 * Authors: Marco Tezzele, Nicola Cavallini, Luca Heltai 2014-2015
 */

/*  This code is a modification of step-4 of the deal.II library,
 *  which was used to produce the results of the article
 *
 *  "Algorithms, data structures and applications for Isogeometric
 *  Analysis with the deal.II library",
 *  Marco Tezzele, Nicola Cavallini, Luca Heltai
 *
 * Usage: ./poisson finite_element_name quadrature_name degree first_cycle last_cycle
 * for example ./poisson bernstein legendre 2 1 3
 *
 * The accepted arguments of the program are
 *
 * finite_element_name  : bernstein, lagrange, lobatto
 * quadrature_name  : legendre, lobatto
 * degree   : degree of the finite element space
 * first_cycle    : initial refinement of the grid
 * last_cycle   : final refinement of the grid
 *
 * Only the differences between this program and step-4 of the deal.II
 * library have been documented. Please see there for a full
 * explanation of the undocumented parts.
 */

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_bernstein.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/convergence_table.h>
#include <fstream>
#include <iostream>
#include <deal.II/base/logstream.h>


using namespace dealii;

template <int dim>
class Solution : public Function<dim>
{
public:
  Solution () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};

template <int dim>
double Solution<dim>::value (const Point<dim>   &p,
                             const unsigned int) const
{
  return std::sin(2*p(0)*numbers::PI)*std::sin(3*p(1)*numbers::PI);
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
                                   const unsigned int /*component*/) const
{
  return std::sin(2*p(0)*numbers::PI)*std::sin(3*p(1)*numbers::PI);
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
                                  const unsigned int /*component*/) const
{
  return 13*std::pow(numbers::PI,2)*std::sin(2*p(0)*numbers::PI)*std::sin(3*p(1)*numbers::PI);
}


template <int dim>
class Laplace
{
public:
  // The constructor differ from the step-4 example of the deal.II
  // library, in that this program admits some parameters from the
  // command line, and they are passed to the constructor of the
  // problem as additional arguments.
  //
  //
  // The accepted arguments of the program are
  //
  // fe_name    : bernstein, lagrange, lobatto
  // quadrature_name  : legendre, lobatto
  // degree   : degree of the finite element space
  // n_cycles_up  : initial refinement of the grid
  // n_cycles_down  : final refinement of the grid
  //
  Laplace (const std::string fe_name,
           const std::string quadrature_name,
           const unsigned int degree,
           const unsigned int n_cycles_low,
           const unsigned int n_cycles_up);
  ~Laplace();
  void run ();

private:
  void make_grid ();
  void setup_system();
  void assemble_system ();
  void solve ();
  void refine_grid ();
  void output_results (const unsigned int cycle) const;
  void process_solution (const unsigned int cycle);

  const std::string fe_name;
  const std::string quadrature_name;
  const unsigned int degree;
  const unsigned int n_cycles_low;
  const unsigned int n_cycles_up;
  unsigned int cg_iter;

  Triangulation<dim>   triangulation;

  // The finite element is a pointer instead of an object, because we
  // don't know it yet at construction time, and FiniteElements do not
  // support assignement operators.

  FiniteElement<dim>   *fe;
  DoFHandler<dim>      dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;

  ConvergenceTable     convergence_table;

  Quadrature<dim>    matrix_quad;
  Quadrature<dim>    error_quad;
  Quadrature<dim-1>  boundary_quad;
};


template <int dim>
Laplace<dim>::Laplace (const std::string fe_name,
                       const std::string quadrature_name,
                       const unsigned int degree,
                       const unsigned int n_cycles_low,
                       const unsigned int n_cycles_up)
  :
  fe_name(fe_name),
  quadrature_name(quadrature_name),
  degree(degree),
  n_cycles_low(n_cycles_low),
  n_cycles_up(n_cycles_up),
  fe(NULL),
  dof_handler (triangulation)
{
  if (quadrature_name == "legendre")
    {
      matrix_quad = QGauss<dim>(degree+1);
      boundary_quad = QGauss<dim-1>(degree+2);
      error_quad = QGauss<dim>(degree+3);
    }
  else if (quadrature_name == "lobatto")
    {
      matrix_quad = QGaussLobatto<dim>(degree+2);
      boundary_quad = QGaussLobatto<dim-1>(degree+3);
      error_quad = QGaussLobatto<dim>(degree+4);
    }
  else
    AssertThrow(false, ExcMessage("Quadrature not Implemented"));

  if (fe_name == "bernstein")
    fe = new FE_Bernstein<dim>(degree);
  else if (fe_name == "lagrange")
    fe = new FE_Q<dim>(degree);
  else if (fe_name == "lobatto")
    fe = new FE_Q<dim>(QGaussLobatto<1>(degree+1));
  else
    AssertThrow(false, ExcMessage("FE not supported"));
}

template <int dim>
Laplace<dim>::~Laplace ()
{
  dof_handler.clear();
  if (fe)
    delete fe;
}


template <int dim>
void Laplace<dim>::make_grid ()
{
  GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global (n_cycles_low+1);
}

template <int dim>
void Laplace<dim>::setup_system ()
{
  dof_handler.distribute_dofs (*fe);

  std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern d_sparsity(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, d_sparsity);
  sparsity_pattern.copy_from(d_sparsity);

  system_matrix.reinit (sparsity_pattern);

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}


template <int dim>
void Laplace<dim>::assemble_system ()
{
  const RightHandSide<dim> right_hand_side;

  FEValues<dim> fe_values (*fe, matrix_quad,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);

  const unsigned int   dofs_per_cell = fe->dofs_per_cell;
  const unsigned int   n_q_points    = matrix_quad.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

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

      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               cell_matrix(i,j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

  std::map<types::global_dof_index,double> boundary_values;

  typename FunctionMap<dim>::type  dirichlet_boundary;

  BoundaryValues<dim> boundary_funct;
  dirichlet_boundary[0] = &boundary_funct;
  VectorTools::project_boundary_values (dof_handler,
                                        dirichlet_boundary,
                                        boundary_quad,
                                        boundary_values);

  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
}


template <int dim>
void Laplace<dim>::solve ()
{
  SolverControl           solver_control (100000, 1e-14);
  SolverCG<>              solver (solver_control);

  std::cout << "   Memory consumption " << system_matrix.memory_consumption()
            << " bytes" << std::endl;

  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence."
            << std::endl;
  cg_iter = solver_control.last_step();
}


template <int dim>
void Laplace<dim>::refine_grid()
{
  triangulation.refine_global (1);
}


template <int dim>
void Laplace<dim>::output_results (const unsigned int cycle) const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");

  data_out.build_patches ();

  std::string filename = "solution-2d-";
  filename += ('0' + cycle);
  filename += ".vtk";

  std::ofstream output (filename.c_str());
  data_out.write_vtk (output);
}


template <int dim>
void Laplace<dim>::process_solution(const unsigned int cycle)
{
  Vector<float> difference_per_cell (triangulation.n_active_cells());
  VectorTools::integrate_difference (dof_handler,
                                     solution,
                                     Solution<dim>(),
                                     difference_per_cell,
                                     error_quad,
                                     VectorTools::L2_norm);
  const double L2_error = difference_per_cell.l2_norm();

  const unsigned int n_active_cells=triangulation.n_active_cells();
  const unsigned int n_dofs=dof_handler.n_dofs();

  convergence_table.add_value("cycle", cycle);
  convergence_table.add_value("cells", n_active_cells);
  convergence_table.add_value("dofs", n_dofs);
  convergence_table.add_value("CG", cg_iter);
  convergence_table.add_value("memory", (unsigned int)system_matrix.memory_consumption());
  convergence_table.add_value("L2", L2_error);
}


template <int dim>
void Laplace<dim>::run ()
{
  std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;

  for (unsigned int cycle = n_cycles_low; cycle < n_cycles_up+1; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;
      if (cycle == n_cycles_low)
        {
          make_grid();
        }
      else
        {
          refine_grid();
        }
      std::cout << "   Number of active cells: "
                << triangulation.n_active_cells()
                << std::endl
                << "   Total number of cells: "
                << triangulation.n_cells()
                << std::endl;
      setup_system ();
      assemble_system ();
      solve ();
      output_results (cycle);
      process_solution (cycle);
    }

  convergence_table.set_precision("L2", 3);
  convergence_table.set_scientific("L2", true);
  convergence_table.set_tex_caption("cells", "\\# cells");
  convergence_table.set_tex_caption("dofs", "\\# dofs");
  convergence_table.set_tex_caption("L2", "$L^2$-error");
  convergence_table.set_tex_format("cells", "r");
  convergence_table.set_tex_format("dofs", "r");
  convergence_table.set_tex_format("CG", "r");
  convergence_table.set_tex_format("memory", "r");
  convergence_table
  .evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
  std::cout << std::endl;
  convergence_table.write_text(std::cout);

  char fname[50], fnametex[50];
  sprintf(fname, "convergence-%s-%s-q%d-r%d-%d.txt",
          fe_name.c_str(), quadrature_name.c_str(),
          fe->degree, n_cycles_low, n_cycles_up);

  sprintf(fnametex, "convergence-%s-%s-q%d-r%d-%d.tex",
          fe_name.c_str(), quadrature_name.c_str(),
          fe->degree, n_cycles_low, n_cycles_up);

  std::ofstream outfile(fname);
  std::ofstream outfiletex(fnametex);
  convergence_table.write_text(outfile);
  convergence_table.write_tex(outfiletex);
}


int main (int argc, char **argv)
{
  char fe_name[] = "bernstein";
  char quad_name[] = "legendre";
  unsigned int degree = 1;
  unsigned int n_cycles_down = 0;
  unsigned int n_cycles_up = 5;

  char *tmp[3];
  tmp[0] = argv[0];
  tmp[1] = fe_name;
  tmp[2] = quad_name;

  if (argc == 1)
    {
      argv = tmp;
    }
  else
    {
      AssertThrow(argc == 6,
                  ExcMessage("Wrong number of arguments: 0 or 5"));
      AssertThrow(sscanf(argv[3], "%d", &degree) == 1,
                  ExcMessage("Unrecognized argument 3"));
      AssertThrow(sscanf(argv[4], "%d", &n_cycles_down) == 1,
                  ExcMessage("Unrecognized argument 4"));
      AssertThrow(sscanf(argv[5], "%d", &n_cycles_up) == 1,
                  ExcMessage("Unrecognized argument 5"));
    }

  deallog.depth_console (0);
  try
    {
      Laplace<2> laplace_problem_2d(argv[1], argv[2], degree, n_cycles_down, n_cycles_up);
      laplace_problem_2d.run ();
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
