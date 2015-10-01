#ifndef __grid_generator_h__
#define __grid_generator_h__

#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table.h>
#include <deal.II/grid/tria.h>
#include <map>

DEAL_II_NAMESPACE_OPEN

template <int dim, int spacedim> class Triangulation;

template <typename number> class Vector;

template <typename number> class SparseMatrix;

namespace GridGenerator
{
  template<int dim, int spacedim=dim>
  void
  subdivided_hyper_rectangle(
    Triangulation<dim,spacedim>              &tria,
    const std::vector< std::vector<double> >   &p);
}

DEAL_II_NAMESPACE_CLOSE
#endif
