#include <iostream>
#include <unistd.h>

#include "DataAdaptor.h"
#include "KombyneAnalysisAdaptor.h"

#include <kombyne_data.h>
#include <kombyne_execution.h>

#include "SVTKUtils.h"

#include <svtkCellData.h>
#include <svtkCharArray.h>
#include <svtkCompositeDataIterator.h>
#include <svtkCompositeDataSet.h>
#include <svtkDataArray.h>
#include <svtkDataSet.h>
#include <svtkDataSetAttributes.h>
#include <svtkImageData.h>
#include <svtkOverlappingAMR.h>
#include <svtkPointData.h>
#include <svtkPolyData.h>
#include <svtkRectilinearGrid.h>
#include <svtkStructuredGrid.h>
#include <svtkUniformGridAMRDataIterator.h>
#include <svtkUnsignedCharArray.h>
#include <svtkUnstructuredGrid.h>


namespace sensei
{

// Note: This adaptor borrows heavily from the LibsimAnalysisAdaptor

// -----------------------------------------------------------------------------
// SVTK to Kombyne helper functions
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
static kb_var_handle svtkDataArrayToKombyneVariable(svtkDataArray *arr)
{
  kb_var_handle h = KB_HANDLE_NULL;
  if (arr != nullptr)
  {
    // If we have a standard memory layout in a supported type,
    // zero-copy expose the data to Kombyne.
    h = kb_var_alloc();
    if (h != KB_HANDLE_NULL)
    {
      kb_return status = KB_RETURN_ERROR;
      bool copy = false;
      int nc = arr->GetNumberOfComponents();
      int nt = arr->GetNumberOfTuples();

      if (arr->HasStandardMemoryLayout())
      {
        if (arr->GetDataType() == SVTK_CHAR)
          status = kb_var_set(h, KB_MEM_BORROW, KB_STORAGE_CHAR,
              nc, nt, arr->GetVoidPointer(0));
        else if (arr->GetDataType() == SVTK_UNSIGNED_CHAR)
          status = kb_var_set(h, KB_MEM_BORROW, KB_STORAGE_UCHAR,
              nc, nt, arr->GetVoidPointer(0));
        else if (arr->GetDataType() == SVTK_INT)
          status = kb_var_set(h, KB_MEM_BORROW, KB_STORAGE_INT,
              nc, nt, arr->GetVoidPointer(0));
        else if (arr->GetDataType() == SVTK_LONG)
          status = kb_var_set(h, KB_MEM_BORROW, KB_STORAGE_LONG,
              nc, nt, arr->GetVoidPointer(0));
        else if (arr->GetDataType() == SVTK_FLOAT)
          status = kb_var_set(h, KB_MEM_BORROW, KB_STORAGE_FLOAT,
              nc, nt, arr->GetVoidPointer(0));
        else if (arr->GetDataType() == SVTK_DOUBLE)
          status = kb_var_set(h, KB_MEM_BORROW, KB_STORAGE_DOUBLE,
              nc, nt, arr->GetVoidPointer(0));
        else
            copy = true;

        if (!copy)
        {
          //SENSEI_STATUS("Standard memory layout: nc=" << nc << ", nt=" << nt)

          if (status != KB_RETURN_OKAY)
          {
            kb_var_free(h);
            h = KB_HANDLE_NULL;
          }
        }
      }
      else
      {
        // NOTE: we could detect some non-contiguous memory layouts
        // here and expose to Kombyne that way. Just copy for now...
        copy = true;
      }

      // Expose the data as a copy, converting to double.
      if (copy)
      {
        //SENSEI_STATUS("Copying required: nc=" << nc << ", nt=" << nt)

        double *v = (double *)malloc(sizeof(double) * nc * nt);
        if (v != nullptr)
        {
          double *tuple = v;
          for (int i = 0; i < nt; ++i)
          {
            arr->GetTuple(i, tuple);
            tuple += nc;
          }
          status = kb_var_set(h, KB_MEM_COPY, KB_STORAGE_DOUBLE, nc, nt, v);
        }
        if (v != nullptr && status != KB_RETURN_OKAY)
        {
          kb_var_free(h);
          h = KB_HANDLE_NULL;
        }
      }
    }
  }
  return h;
}

// -----------------------------------------------------------------------------
static int svtkToKombyneCellType(unsigned char svtkcelltype)
{
  static int celltypeMap[SVTK_NUMBER_OF_CELL_TYPES];
  static bool celltypeMapInitialized = false;

  if (!celltypeMapInitialized)
  {
    for (int i =0; i < SVTK_NUMBER_OF_CELL_TYPES; ++i)
      celltypeMap[i] = -1;

    celltypeMap[SVTK_LINE] = KB_CELLTYPE_EDGE;
    celltypeMap[SVTK_TRIANGLE] = KB_CELLTYPE_TRI;
    celltypeMap[SVTK_QUAD] = KB_CELLTYPE_QUAD;
    celltypeMap[SVTK_TETRA] = KB_CELLTYPE_TET;
    celltypeMap[SVTK_PYRAMID] = KB_CELLTYPE_PYR;
    celltypeMap[SVTK_WEDGE] = KB_CELLTYPE_WEDGE;
    celltypeMap[SVTK_HEXAHEDRON] = KB_CELLTYPE_HEX;
    celltypeMap[SVTK_VERTEX] = KB_CELLTYPE_VERTEX;

    celltypeMapInitialized = true;
  }
  return celltypeMap[svtkcelltype];
}

// -----------------------------------------------------------------------------
static kb_var_handle svtkDataSet_GhostData(
    svtkDataSetAttributes *dsa, const std::string &name)
{
  kb_var_handle h = KB_HANDLE_NULL;
  // Check that we have the array and it is of allowed types.
  svtkDataArray *arr = dsa->GetArray(name.c_str());
  if (arr &&
     arr->GetNumberOfComponents() == 1 &&
     arr->GetNumberOfTuples() > 0 &&
     (svtkUnsignedCharArray::SafeDownCast(arr) ||
      svtkCharArray::SafeDownCast(arr) ||
      svtkIntArray::SafeDownCast(arr))
    )
  {
    h = svtkDataArrayToKombyneVariable(arr);
  }
  return h;
}

// -----------------------------------------------------------------------------
static kb_fields_handle svtkDataSet_Variables(
    svtkDataObject *dobj, MeshMetadataPtr mdptr)
{
  kb_fields_handle hfields = kb_fields_alloc();

  if (hfields != KB_HANDLE_NULL)
  {
    // TODO: support subsetting variable list
    for (int j = 0; j < mdptr->NumArrays; ++j)
    {
      std::string varName = mdptr->ArrayName[j];
      int centering = mdptr->ArrayCentering[j];

      svtkDataArray *array = nullptr;
      array = dobj->GetAttributes(centering)->GetArray(varName.c_str());
      if (array == nullptr)
      {
        SENSEI_ERROR("Failed to get variable \"" << varName << "\"")
        continue;
      }

      kb_var_handle hvar = svtkDataArrayToKombyneVariable(array);
      if (hvar == KB_HANDLE_NULL)
      {
        SENSEI_ERROR("Failed to create Kombyne variable \"" << varName << "\"")
        continue;
      }

      kb_centering vc = (centering == svtkDataObject::POINT) ?
        KB_CENTERING_POINTS : KB_CENTERING_CELLS;

      if (varName == "iblank")
        kb_fields_add_var(hfields, "kb_iblank", vc, hvar);
      else
        kb_fields_add_var(hfields, varName.c_str(), vc, hvar);
    }
  }
  return hfields;
}

// -----------------------------------------------------------------------------
static kb_mesh_handle svtkDataSet_Mesh(
    svtkDataObject *dobj, MeshMetadataPtr mdptr)
{
  svtkDataSet *ds = dynamic_cast<svtkDataSet*>(dobj);
  if (dobj && !ds)
  {
    SENSEI_ERROR("Can't convert a "
        << (dobj ? dobj->GetClassName() : "nullptr")
        << " to a Kombyne mesh.")
    return KB_HANDLE_NULL;
  }

  kb_mesh_handle mesh = KB_HANDLE_NULL;
  svtkImageData *igrid = svtkImageData::SafeDownCast(ds);
  svtkRectilinearGrid *rgrid = svtkRectilinearGrid::SafeDownCast(ds);
  svtkStructuredGrid *sgrid = svtkStructuredGrid::SafeDownCast(ds);
  svtkPolyData *pgrid = svtkPolyData::SafeDownCast(ds);
  svtkUnstructuredGrid *ugrid = svtkUnstructuredGrid::SafeDownCast(ds);

  if (igrid != nullptr)
  {
    //SENSEI_STATUS("Exposing svtkImageData as a rectilinear grid.")

    double x0[3] = {0.0};
    double dx[3] = {0.0};
    int dims[3] = {0};
    int ext[6] = {0};
    igrid->GetDimensions(dims);
    igrid->GetExtent(ext);
    igrid->GetOrigin(x0);
    igrid->GetSpacing(dx);

    auto hrgrid = kb_rgrid_alloc();

    if (hrgrid != KB_HANDLE_NULL)
    {
      int nx = std::max(dims[0], 1);
      int ny = std::max(dims[1], 1);
      int nz = std::max(dims[2], 1);

      kb_rgrid_set_dims(hrgrid, dims);

      float *x = (float *)malloc(sizeof(float) * nx);
      float *y = (float *)malloc(sizeof(float) * ny);
      float *z = (float *)malloc(sizeof(float) * nz);

      if (x != nullptr && y != nullptr && z != nullptr)
      {
        auto hx = kb_var_alloc();
        auto hy = kb_var_alloc();
        auto hz = kb_var_alloc();

        if (hx != KB_HANDLE_NULL &&
            hy != KB_HANDLE_NULL &&
            hz != KB_HANDLE_NULL)
        {
          for (int i = 0; i < nx; ++i)
            x[i] = x0[0] + (ext[0] + i) * dx[0];
          kb_var_setf(hx, KB_MEM_COPY, 1, nx, x);
          for (int i = 0; i < ny; ++i)
            y[i] = x0[1] + (ext[1] + i) * dx[1];
          kb_var_setf(hy, KB_MEM_COPY, 1, ny, y);
          if (nz > 1)
          {
            for (int i = 0; i < ny; ++i)
              z[i] = x0[2] + (ext[2] + i) * dx[2];
            kb_var_setf(hz, KB_MEM_COPY, 1, nz, z);
          }
          else
          {
            free(z);
            z = nullptr;
          }
          kb_rgrid_set_coords(hrgrid, hx, hy, hz);

          // Try and make some ghost nodes.
          kb_var_handle gn = svtkDataSet_GhostData(
              ds->GetPointData(), "svtkGhostType");
          if (gn != KB_HANDLE_NULL)
            kb_rgrid_set_ghost_nodes(hrgrid, gn);

          // Try and make some ghost cells.
          kb_var_handle gc = svtkDataSet_GhostData(
              ds->GetCellData(), "svtkGhostType");
          if (gc != KB_HANDLE_NULL)
            kb_rgrid_set_ghost_cells(hrgrid, gc);

          kb_rgrid_set_fields(hrgrid, svtkDataSet_Variables(dobj, mdptr));
          mesh = (kb_mesh_handle) hrgrid;
        }
        else
        {
          if (hx != KB_HANDLE_NULL)
            kb_var_free(hx);
          if (hy != KB_HANDLE_NULL)
            kb_var_free(hy);
          if (hz != KB_HANDLE_NULL)
            kb_var_free(hz);
          if (x != nullptr)
            free(x);
          if (y != nullptr)
            free(y);
          if (z != nullptr)
            free(z);
          kb_rgrid_free(hrgrid);
        }
      }
      else
      {
        if (x != nullptr)
          free(x);
        if (y != nullptr)
          free(y);
        if (z != nullptr)
          free(z);
        kb_rgrid_free(hrgrid);
      }
    }
  }
  else if (rgrid != nullptr)
  {
    auto hrgrid = kb_rgrid_alloc();

    if (hrgrid != KB_HANDLE_NULL)
    {
      kb_var_handle hx, hy, hz;
      hx = svtkDataArrayToKombyneVariable(rgrid->GetXCoordinates());
      hy = svtkDataArrayToKombyneVariable(rgrid->GetYCoordinates());
      hz = svtkDataArrayToKombyneVariable(rgrid->GetZCoordinates());
      if (hz == KB_HANDLE_NULL)
        hz = kb_var_alloc();

      if (hx != KB_HANDLE_NULL && hy != KB_HANDLE_NULL && hz != KB_HANDLE_NULL)
      {
        kb_rgrid_set_coords(hrgrid, hx, hy, hz);

        // Try and make some ghost nodes.
        kb_var_handle gn = svtkDataSet_GhostData(
            ds->GetPointData(), "svtkGhostType");
        if (gn != KB_HANDLE_NULL)
          kb_rgrid_set_ghost_nodes(hrgrid, gn);

        // Try and make some ghost cells.
        kb_var_handle gc = svtkDataSet_GhostData(
            ds->GetCellData(), "svtkGhostType");
        if (gc != KB_HANDLE_NULL)
          kb_rgrid_set_ghost_cells(hrgrid, gc);

        kb_rgrid_set_fields(hrgrid, svtkDataSet_Variables(dobj, mdptr));
        mesh = (kb_mesh_handle) hrgrid;
      }
      else
      {
        if (hx != KB_HANDLE_NULL)
          kb_var_free(hx);
        if (hy != KB_HANDLE_NULL)
          kb_var_free(hy);
        if (hz != KB_HANDLE_NULL)
          kb_var_free(hz);
        kb_rgrid_free(hrgrid);
      }
    }
  }
  else if (sgrid != nullptr)
  {
    auto hsgrid = kb_sgrid_alloc();

    if (hsgrid != KB_HANDLE_NULL)
    {
      int dims[3];
      sgrid->GetDimensions(dims);
      kb_var_handle pts = svtkDataArrayToKombyneVariable(
          sgrid->GetPoints()->GetData());
      if (pts != KB_HANDLE_NULL)
      {
        kb_sgrid_set_coords(hsgrid, pts);

        // Try and make some ghost nodes.
        kb_var_handle gn = svtkDataSet_GhostData(
            ds->GetPointData(), "svtkGhostType");
        if (gn != KB_HANDLE_NULL)
          kb_sgrid_set_ghost_nodes(hsgrid, gn);

        // Try and make some ghost cells.
        kb_var_handle gc = svtkDataSet_GhostData(
            ds->GetCellData(), "svtkGhostType");
        if (gc != KB_HANDLE_NULL)
          kb_sgrid_set_ghost_cells(hsgrid, gc);

        kb_sgrid_set_fields(hsgrid, svtkDataSet_Variables(dobj, mdptr));
        mesh = (kb_mesh_handle) hsgrid;
      }
      else
        kb_sgrid_free(hsgrid);
    }
  }
  else if (pgrid && pgrid->GetVerts())
  {
    auto hugrid = kb_ugrid_alloc();

    if (hugrid != KB_HANDLE_NULL)
    {
      bool err = false;
      kb_var_handle pts = svtkDataArrayToKombyneVariable(
          pgrid->GetPoints()->GetData());
      if (pts != KB_HANDLE_NULL)
        kb_ugrid_set_coords(hugrid, pts);
      else
        err = true;

      svtkIdType ncells = pgrid->GetNumberOfPoints();

      kb_var_handle hc = kb_var_alloc();
      if (hc != KB_HANDLE_NULL)
      {
        int *newconn = (int *) malloc(sizeof(int) * ncells);

        if (newconn != nullptr)
        {
          for (int i = 0; i < ncells; ++i)
            newconn[i] = i;

          // Wrap newconn, let Kombyne own it.
          kb_var_seti(hc, KB_MEM_COPY, 1, ncells, newconn);
          kb_ugrid_add_cells(hugrid, KB_CELLTYPE_VERTEX, hc);
          kb_ugrid_set_fields(hugrid, svtkDataSet_Variables(dobj, mdptr));
          mesh = (kb_mesh_handle) hugrid;
        }
	err = true;
      }

      if (err)
	kb_ugrid_free(hugrid);
    }
  }
  else if (ugrid != nullptr)
  {
    //SENSEI_STATUS("svtkUnstructuredGrid: npts = "
    //    << ugrid->GetNumberOfPoints() << ", ncells = "
    //    << ugrid->GetNumberOfCells())

    svtkIdType ncells = ugrid->GetNumberOfCells();
    if (ncells > 0)
    {
      auto hugrid = kb_ugrid_alloc();

      if (hugrid != KB_HANDLE_NULL)
      {
        bool err = false;

        kb_var_handle pts = svtkDataArrayToKombyneVariable(
            ugrid->GetPoints()->GetData());
        if (pts != KB_HANDLE_NULL)
          kb_ugrid_set_coords(hugrid, pts);
        else
          err = true;

        const unsigned char *cellTypes = (const unsigned char *)
          ugrid->GetCellTypesArray()->GetVoidPointer(0);
        const svtkIdType *svtkconn = (const svtkIdType *)
          ugrid->GetCells()->GetConnectivityArray()->GetVoidPointer(0);
        const svtkIdType *offsets = (const svtkIdType *)
          ugrid->GetCells()->GetOffsetsArray()->GetVoidPointer(0);

        int connlen = ugrid->GetCells()->GetNumberOfConnectivityEntries();
        int *newconn = (int *) malloc(sizeof(int) * connlen);

        if (newconn == nullptr)
          err = true;
        else
        {
          int *lsconn = newconn;

          for (int cellid = 0; cellid < ncells; ++cellid)
          {
            // Map SVTK cell type to Kombyne cell type.
            int lsct = svtkToKombyneCellType(cellTypes[cellid]);
            if (lsct != -1)
            {
              *lsconn++ = lsct;

              // The number of points is the first number for the cell.
              const svtkIdType *cellConn = svtkconn + offsets[cellid];
              svtkIdType npts = offsets[cellid + 1] - offsets[cellid];
              for (svtkIdType idx = 0; idx < npts; ++idx)
                *lsconn++ = static_cast<int>(cellConn[idx]);
            }
            else
            {
              // We got a cell type we don't support. Make a vertex cell
              // so we at least don't mess up the cell data later.
              *lsconn++ = KB_CELLTYPE_VERTEX;
              const svtkIdType *cellConn = svtkconn + offsets[cellid];
              *lsconn++ = cellConn[0];
            }
          }

          kb_var_handle hc = kb_var_alloc();
          if (hc != KB_HANDLE_NULL)
          {
            // Wrap newconn, let Kombyne own it.
            kb_var_seti(hc, KB_MEM_COPY, 1, connlen, newconn);

            kb_ugrid_add_cells_interleaved(hugrid, hc);

            // Try and make some ghost nodes.
            kb_var_handle gn = svtkDataSet_GhostData(
                ds->GetPointData(), "svtkGhostType");
            if (gn != KB_HANDLE_NULL)
              kb_ugrid_set_ghost_nodes(hugrid, gn);

            // Try and make some ghost cells.
            kb_var_handle gc = svtkDataSet_GhostData(
                ds->GetCellData(), "svtkGhostType");
            if (gc != KB_HANDLE_NULL)
              kb_ugrid_set_ghost_cells(hugrid, gc);

            kb_ugrid_set_fields(hugrid, svtkDataSet_Variables(dobj, mdptr));
            mesh = (kb_mesh_handle) hugrid;
          }
          else
          {
            free(newconn);
            err = true;
          }
        }

        if (err)
          kb_ugrid_free(hugrid);
      }
    }
  }
  // TODO: expand to other mesh types.
  else
  {
    SENSEI_ERROR("Unsupported SVTK mesh type "
        << (ds ? ds->GetClassName() : dobj ? dobj->GetClassName() : "nullptr"))
  }
  return mesh;
}


// -----------------------------------------------------------------------------
// KombyneAnalysisAdaptor API
// -----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
senseiNewMacro(KombyneAnalysisAdaptor);

//-----------------------------------------------------------------------------
KombyneAnalysisAdaptor::KombyneAnalysisAdaptor() :
  Adaptor(nullptr),
  pipelineFile("kombyne.yaml"),
  sessionName("kombyne-session"),
  role(KB_ROLE_SIMULATION_AND_ANALYSIS),
  verbose(false),
  initialized(false),
  hp(KB_HANDLE_NULL)
{
}

//-----------------------------------------------------------------------------
KombyneAnalysisAdaptor::~KombyneAnalysisAdaptor()
{
  if (this->hp != KB_HANDLE_NULL)
    kb_pipeline_collection_free(this->hp);
}

//-----------------------------------------------------------------------------
void KombyneAnalysisAdaptor::Initialize()
{
  int rank, size;

  MPI_Comm comm = this->GetCommunicator();
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  kb_role newrole;  // will be the same as the input role
  MPI_Comm split;   // will be the same as the input communicator

  kb_initialize(comm,
      "KombyneAnalysisAdaptor",
      "Kombyne analysis adaptor for SENSEI",
      this->role,
      size, size,
      this->sessionName.c_str(),
      &split,
      &newrole);

  // Create a pipeline collection
  this->hp = kb_pipeline_collection_alloc();
  if (this->hp == KB_HANDLE_NULL)
  {
    SENSEI_ERROR("Failed to allocate Kombyne pipeline collection.");
    return;
  }

  kb_pipeline_collection_set_filename(this->hp, this->pipelineFile.c_str());
  if (kb_pipeline_collection_initialize(this->hp) != KB_RETURN_OKAY)
  {
    SENSEI_ERROR("Failed to initialize Kombyne pipeline collection.");
    kb_pipeline_collection_free(this->hp);
    this->hp = KB_HANDLE_NULL;
    return;
  }

  this->initialized = true;
}

// --------------------------------------------------------------------------
int KombyneAnalysisAdaptor::GetMetaData(void)
{
  // for each mesh we'll pass metadata onto Kombyne
  unsigned int nMeshes = 0;
  if (Adaptor->GetNumberOfMeshes(nMeshes))
  {
    SENSEI_ERROR("Failed to get the number of meshes")
    return -1;
  }

  // set up the metadata cache
  this->Metadata.clear();

  for (unsigned int i = 0; i < nMeshes; ++i)
  {
    MeshMetadataPtr mmd = MeshMetadata::New();

    // enable optional metadata
    mmd->Flags.SetBlockDecomp();
    mmd->Flags.SetBlockExtents();

    if (Adaptor->GetMeshMetadata(i, mmd))
    {
      SENSEI_ERROR("Failed to get metadata for mesh " << i)
      return -1;
    }

    // check if the sim gave us what we asked for
    MeshMetadataFlags reqFlags;
    reqFlags.SetBlockDecomp();
    reqFlags.SetBlockExtents();

    if (mmd->Validate(this->GetCommunicator(), reqFlags))
    {
      SENSEI_ERROR("Invalid metadata for mesh " << i)
      return -1;
    }

    // this simplifies things substantially to be able to have a global view
    // the driver behind this is AMR data, for which we require a global view.
    if (!mmd->GlobalView)
      mmd->GlobalizeView(this->GetCommunicator());

    // cache the metadata
    this->Metadata[mmd->MeshName] = mmd;
  }

  return 0;
}

// --------------------------------------------------------------------------
int KombyneAnalysisAdaptor::GetMesh(
    const std::string &meshName, svtkDataObjectPtr &dobjp)
{
    dobjp = nullptr;

    // get the mesh. it's cached because Kombyne wants things block
    // by block but sensei only works with the whole object
    auto it = this->Meshes.find(meshName);
    if (it  == this->Meshes.end())
    {
        // mesh was not in the cache add it now
        svtkDataObject *dobj = nullptr;
        if (this->Adaptor->GetMesh(meshName, false, dobj))
        {
            SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
            return -1;
        }

        // get the metadata, it should already be available
        auto mdit = this->Metadata.find(meshName);
        if (mdit == this->Metadata.end())
        {
            SENSEI_ERROR("No metadata for mesh \"" << meshName << "\"")
            return -1;
        }
        MeshMetadataPtr mmd = mdit->second;

        // add ghost zones. if the simulation has them we always want/need
        // them
        if ((mmd->NumGhostCells || SVTKUtils::AMR(mmd)) &&
          this->Adaptor->AddGhostCellsArray(dobj, meshName))
        {
            SENSEI_ERROR("Failed to add ghost cells to mesh \""
              << meshName << "\"")
            return -1;
        }

        if (mmd->NumGhostNodes &&
          this->Adaptor->AddGhostNodesArray(dobj, meshName))
        {
            SENSEI_ERROR("Failed to add ghost nodes to mesh \""
              << meshName << "\"")
            return -1;
        }

        dobjp.TakeReference(dobj);
        this->Meshes[meshName] = dobjp;
    }
    else
        dobjp = it->second;

    return 0;
}

//-----------------------------------------------------------------------------
svtkDataObject *KombyneAnalysisAdaptor::GetMeshBlock(
    MPI_Comm comm, const int domain, MeshMetadataPtr mdptr)
{
  svtkDataObject *mesh = nullptr;

  svtkDataObjectPtr wmesh;
  if (this->GetMesh(mdptr->MeshName, wmesh))
  {
    SENSEI_ERROR("Failed to get simulation mesh.");
    return nullptr;
  }
  svtkCompositeDataSetPtr cd = SVTKUtils::AsCompositeData(comm, wmesh, false);

  // get the block that Kombyne is after
  svtkCompositeDataIterator *cdit = cd->NewIterator();

  if (! cdit->IsDoneWithTraversal())
  {
    // TODO: support subsetting variable list
    for (int j = 0; j < mdptr->NumArrays; ++j)
    {
      std::string varName = mdptr->ArrayName[j];
      int centering = mdptr->ArrayCentering[j];

      // read the array if we have not yet
      if (! cdit->GetCurrentDataObject()->GetAttributes(
          centering)->GetArray(varName.c_str()))
      {
        if (this->Adaptor->AddArray(
              wmesh.GetPointer(), mdptr->MeshName, centering, varName))
        {
          SENSEI_ERROR("Failed to add "
              << SVTKUtils::GetAttributesName(centering)
              << " data array \"" << varName << "\"")
          cdit->Delete();
          continue;
        }
      }
    }
  }

  // extract array from the requested block

  // SVTK's iterators for AMR datasets behave differently than for multiblock
  // datasets.  we are going to have to handle AMR data as a special case for
  // now.

  svtkUniformGridAMRDataIterator *amrIt =
    dynamic_cast<svtkUniformGridAMRDataIterator*>(cdit);
  svtkOverlappingAMR *amrMesh = dynamic_cast<svtkOverlappingAMR*>(cd.Get());

  for (cdit->InitTraversal(); !cdit->IsDoneWithTraversal(); cdit->GoToNextItem())
  {
      long blockId = 0;
      if (amrIt)
      {
          // special case for AMR
          int level = amrIt->GetCurrentLevel();
          int index = amrIt->GetCurrentIndex();
          blockId = amrMesh->GetAMRBlockSourceIndex(level, index);
      }
      else
      {
          // other composite data
          blockId = cdit->GetCurrentFlatIndex() - 1;
      }

      if (blockId == domain)
      {
          mesh = cdit->GetCurrentDataObject();
          break;
      }
  }

  cdit->Delete();

  if (mesh == nullptr)
  {
      SENSEI_ERROR("Failed to get domain " << domain << " from mesh \""
          << mdptr->MeshName << "\"")
  }

  return mesh;
}

//-----------------------------------------------------------------------------
void KombyneAnalysisAdaptor::ClearCache()
{
  this->Meshes.clear();
  this->Metadata.clear();
}

//-----------------------------------------------------------------------------
/// Invoke in situ processing using Kombyne
bool KombyneAnalysisAdaptor::Execute(DataAdaptor* data, DataAdaptor** dataOut)
{
  this->Adaptor = data;

  // We don't currently return any data.
  if (dataOut)
    dataOut = nullptr;

  kb_pipeline_data_handle hpd;

  if (this->hp == KB_HANDLE_NULL)
    return false; // we already printed an error during initialization

  // Assemble a pipeline_data
  hpd = kb_pipeline_data_alloc();
  if (hpd == KB_HANDLE_NULL)
  {
    SENSEI_ERROR("Failed to allocate Kombyne pipeline data.");
    return false;
  }

  // Expose simulation data to Kombyne here
  this->GetMetaData();

  bool staticMesh = true;
  for (auto mdit = this->Metadata.begin(); mdit != this->Metadata.end(); mdit++)
  {
    MeshMetadataPtr mmd = mdit->second;
    if (! mmd->StaticMesh)
    {
      staticMesh = false;
      break;
    }
  }

  int promises = 0; //KB_PROMISE_STATIC_FIELDS;
  if (staticMesh)
    promises |= KB_PROMISE_STATIC_GRID;

  kb_pipeline_data_set_promises(hpd, promises);

  MPI_Comm comm = this->GetCommunicator();
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int timestep = Adaptor->GetDataTimeStep();
  double time = Adaptor->GetDataTime();

  unsigned int nmesh;
  if (data->GetNumberOfMeshes(nmesh))
  {
    SENSEI_ERROR("Failed to get simulation mesh info.");
    return false;
  }

  // TODO: calculate during GetMetaData
  int numdom = 0;
  for (unsigned int i = 0; i < nmesh; i++)
  {
    MeshMetadataPtr mdptr = MeshMetadata::New();
    if (data->GetMeshMetadata(i, mdptr))
    {
      SENSEI_ERROR("Failed to get simulation mesh info.");
      return false;
    }
    mdptr = this->Metadata[mdptr->MeshName];

    numdom += mdptr->NumBlocks;
  }

  for (unsigned int i = 0; i < nmesh; i++)
  {
    MeshMetadataPtr mdptr = MeshMetadata::New();
    if (data->GetMeshMetadata(i, mdptr))
    {
      SENSEI_ERROR("Failed to get simulation mesh info.");
      return false;
    }
    mdptr = this->Metadata[mdptr->MeshName];

    for (int j = 0; j < mdptr->NumBlocks; ++j)
    {
      if (mdptr->BlockOwner[j] != rank)
        continue;

      int domain = mdptr->BlockIds[j];

      svtkDataObject *mesh = GetMeshBlock(comm, domain, mdptr);
      if (mesh == nullptr)
      {
        SENSEI_ERROR("Failed to get simulation mesh.");
        return false;
      }

      kb_mesh_handle hmesh = svtkDataSet_Mesh(mesh, mdptr);
      if (hmesh != KB_HANDLE_NULL)
      {
        // Add mesh to pipeline_data
        kb_pipeline_data_add(hpd, domain, numdom, timestep, time, hmesh);
      }
    }
  }

  // Execute the simulation side of things.
  kb_simulation_execute(this->hp, hpd, nullptr);

  // Free the pipeline data.
  kb_pipeline_data_free(hpd);

  // During execution data and metadata are cached due to the differnece
  // between how sensei presents data and Kombyne consumes it.
  // You must clear the cache after each execute.
  ClearCache();

  return true;
}

//-----------------------------------------------------------------------------
/// Shuts Kombyne down.
int KombyneAnalysisAdaptor::Finalize()
{
  kb_finalize();
  this->initialized = false;
  return 0;
}

//-----------------------------------------------------------------------------
int KombyneAnalysisAdaptor::SetPipelineFile(std::string filename)
{
  this->pipelineFile = filename;
  return 0;
}

//-----------------------------------------------------------------------------
int KombyneAnalysisAdaptor::SetSessionName(std::string sessionname)
{
  this->sessionName = sessionname;
  return 0;
}

//-----------------------------------------------------------------------------
int KombyneAnalysisAdaptor::SetMode(std::string mode)
{
  if (mode == "in-transit")
    this->role = KB_ROLE_SIMULATION;
  else if (mode == "in-situ")
    this->role = KB_ROLE_SIMULATION_AND_ANALYSIS;
  else
  {
    SENSEI_ERROR("Unknown mode: \"" << mode << "\".  "
        "Expected in-situ or in-transit.");
    return -1;
  }
  return 0;
}

//-----------------------------------------------------------------------------
void KombyneAnalysisAdaptor::SetVerbose(int verbose)
{
  this->verbose = (verbose != 0);
}

}
