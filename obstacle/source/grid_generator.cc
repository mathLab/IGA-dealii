#include "grid_generator.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/lac/filtered_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/lac/filtered_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_reordering.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/matrix_tools.h>

#include <iostream>
#include <cmath>
#include <limits>


DEAL_II_NAMESPACE_OPEN

namespace GridGenerator
{


  template <int dim, int spacedim>
  void
  subdivided_hyper_rectangle(
    Triangulation<dim,spacedim>              &tria,
    const std::vector< std::vector<double> >   &p)
  {
    AssertDimension(p.size(),dim);




    // Check the consistency of
    // the points vector, i.e. that
    // p[d][0] < p[d][1] < ... < p[d][n]
    // for d = 0:dim-1
    //
    for (unsigned int d=0; d<dim; ++d)
      for (unsigned int i=1; i<p[d].size(); ++i)
        Assert(p[d][i-1] < p[d][i],
               ExcMessage("Points are not properly ordered") );



    // then generate the necessary
    // points
    std::vector<Point<spacedim> > points;



    switch (dim)
      {

      case 1:

      {

        for (unsigned int i=0; i<p[0].size(); ++i)
          {
            Point<spacedim> tmp;
            tmp[0] = p[0][i];
            points.push_back( tmp );
          }


        break;

      }


      case 2:

      {

        unsigned int n_x = p[0].size();

        unsigned int n_y = p[1].size();

        for (unsigned int j=0; j<n_y; ++j)
          for (unsigned int i=0; i<n_x; ++i)
            {
              Point<spacedim> tmp;
              tmp[0] = p[0][i];
              tmp[1] = p[1][j];
              points.push_back( tmp );
            }

        break;

      }


      case 3:

      {

        unsigned int n_x = p[0].size();

        unsigned int n_y = p[1].size();

        unsigned int n_z = p[2].size();

        for (unsigned int k=0; k<n_z; ++k)
          for (unsigned int j=0; j<n_y; ++j)
            for (unsigned int i=0; i<n_x; ++i)
              {
                Point<spacedim> tmp;
                tmp[0] = p[0][i];
                tmp[1] = p[1][j];
                tmp[2] = p[2][k];
                points.push_back( tmp );
              }
        break;


      }


      default:

        Assert (false, ExcNotImplemented());

      }



    // next create the cells
    // Prepare cell data
    std::vector<CellData<dim> > cells;

    switch (dim)
      {

      case 1:

      {

        unsigned int x_cells = int(p[0].size())-1;

        cells.resize(x_cells);

        for (unsigned int x=0; x<x_cells; ++x)
          {

            cells[x].vertices[0] = x;

            cells[x].vertices[1] = x+1;

            cells[x].material_id = 0;

          }

        break;

      }


      case 2:

      {

        unsigned int x_cells = int(p[0].size())-1;

        unsigned int y_cells = int(p[1].size())-1;

        cells.resize(x_cells*y_cells);

        for (unsigned int y=0; y<y_cells; ++y)
          for (unsigned int x=0; x<x_cells; ++x)
            {

              const unsigned int c = x+y*(x_cells);

              cells[c].vertices[0] = y*(x_cells+1)+x;

              cells[c].vertices[1] = y*(x_cells+1)+x+1;

              cells[c].vertices[2] = (y+1)*(x_cells+1)+x;

              cells[c].vertices[3] = (y+1)*(x_cells+1)+x+1;

              cells[c].material_id = 0;

            }

        break;

      }


      case 3:

      {

        unsigned int x_cells = int(p[0].size())-1;

        unsigned int y_cells = int(p[1].size())-1;

        unsigned int z_cells = int(p[2].size())-1;

        cells.resize(x_cells*y_cells*z_cells);


        const unsigned int n_x  = (x_cells+1);

        const unsigned int n_xy = (x_cells+1)*(y_cells+1);


        for (unsigned int z=0; z<z_cells; ++z)
          for (unsigned int y=0; y<y_cells; ++y)
            for (unsigned int x=0; x<x_cells; ++x)
              {

                const unsigned int c = x+y*x_cells +
                                       z*x_cells*y_cells;

                cells[c].vertices[0] = z*n_xy + y*n_x + x;

                cells[c].vertices[1] = z*n_xy + y*n_x + x+1;

                cells[c].vertices[2] = z*n_xy + (y+1)*n_x + x;

                cells[c].vertices[3] = z*n_xy + (y+1)*n_x + x+1;

                cells[c].vertices[4] = (z+1)*n_xy + y*n_x + x;

                cells[c].vertices[5] = (z+1)*n_xy + y*n_x + x+1;

                cells[c].vertices[6] = (z+1)*n_xy + (y+1)*n_x + x;

                cells[c].vertices[7] = (z+1)*n_xy + (y+1)*n_x + x+1;

                cells[c].material_id = 0;

              }

        break;


      }


      default:

        Assert (false, ExcNotImplemented());

      }


    tria.create_triangulation (points, cells, SubCellData());
  }


// GridGenerator::shape_to_triangulation(Triangulation<2,3> &tria,
//              const TopoDSShape &shape,
//              bool eliminate_double_vertices=true,
//              double tol=1e-12)
// {

//           //the vertices and cells vectors
//   std::vector<Point<3> > ref_vertices;

//   std::vector<CellData<2> > ref_cells;

//   SubCellData ref_subcelldata;


//           // this is to loop on the faces contained in the shape
//   TopExp_Explorer faceExplorer(sh, TopAbs_FACE);

//   TopoDS_Face face;

//   unsigned int face_count = 0;

//   while(faceExplorer.More())
//     {

//       face = TopoDS::Face(faceExplorer.Current());

//       Standard_Real umin, umax, vmin, vmax;

//       BRepTools::UVBounds(face, umin, umax, vmin, vmax);
//               // create surface
//       Handle(Geom_Surface) surf=BRep_Tool::Surface(face);
//               // get surface associated with face
//               // creating a 2d triangulation here with a single cell here with points located on the umin, umax, vmin, vmax boundaries
//       Triangulation<2,3> ref_triangulation;


//       vertices.push_back(surf.D0(umin,vmin));

//       vertices.push_back(surf.D0(umax,vmin));

//       vertices.push_back(surf.D0(umin,vmax));

//       vertices.push_back(surf.D0(umax,vmax));



//       faceExplorer.Next();

//       ++face_count;

//               //cout<<"Face count: "<<face_count<<endl;
//     }


//           // check
//   AssertThrow(face_count*4 == vertices.size(),
//        ExcMessage("Something is odd with the topology of the CAD faces: number of vertices is not 4 times the number of faces"));

//           // we know we'll have on cell per CAD surface
//   cells.resize(face_count);

//           //vertices are sorted cell by cell, so the connectivity is quite obvious
//   for (unsigned int i=0; i<face_count; ++i)
//     {

//       cells[i].vertices[0]=vertices[i*4];

//       cells[i].vertices[1]=vertices[i*4+1];

//       cells[i].vertices[2]=vertices[i*4+2];

//       cells[i].vertices[3]=vertices[i*4+3];

//               // all cells will have a different material id, associated
//               // as their corresponding CAD face
//       cells[i].material_id = i;

//     }


//           // we now want to join duplicated vertices
//   std::vector<unsigned int> considered_vertices(vertices.size());

//   for (unsigned int i=0; i<vertices.size(); ++i)
//     considered_vertices[i] = i;


//   GridTools::delete_duplicated_vertices(vertices,
//          cells,
//          subcelldata,
//          considered_vertices);



//   GridTools::delete_unused_vertices(vertices, cells, subcelldata);

//   GridReordering<2,3>::reorder_cells(cells);


//   triangulation.create_triangulation_compatibility(vertices, cells, subcelldata );



// }



  template void subdivided_hyper_rectangle<1>(Triangulation<1> &, const std::vector<std::vector<double> > &);
  template void subdivided_hyper_rectangle<2>(Triangulation<2> &, const std::vector<std::vector<double> > &);
  template void subdivided_hyper_rectangle<3>(Triangulation<3> &, const std::vector<std::vector<double> > &);
  template void subdivided_hyper_rectangle<1,2>(Triangulation<1,2> &, const std::vector<std::vector<double> > &);
  template void subdivided_hyper_rectangle<2,3>(Triangulation<2,3> &, const std::vector<std::vector<double> > &);

}

DEAL_II_NAMESPACE_CLOSE
