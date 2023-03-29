/**
 *
 */

// self
#include <OSPRayAnalysisAdaptor.h>
#include "ppmwriter.h"

// stdlib
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// OSPRay
#include <ospray/ospray.h>
#include <ospray/ospray_util.h>

// SENSEI
#include <DataAdaptor.h>
#include <Error.h>

// VTK
#include <svtkCellData.h>
#include <svtkDataObject.h>
#include <svtkDoubleArray.h>
#include <svtkFloatArray.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkPointData.h>
#include <svtkPolyData.h>
#include <svtkUniformGrid.h>
#include <svtkUniformGridAMR.h>
#include <svtkUnsignedCharArray.h>
#include <svtkUnstructuredGrid.h>

#ifdef ENABLE_VTK_FILTERS_PARALLEL_GEOMETRY
#include <SVTKUtils.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPDistributedDataFilter.h>
#include <vtkPKdTree.h>
#endif

namespace sensei {

struct OSPRayAnalysisAdaptor::InternalsType : public svtkObject {
  static InternalsType *New();

  char const *Initialize();
  char const *Execute();
  char const *Finalize();

  svtkSetMacro(Communicator, MPI_Comm);
  svtkSetMacro(Directory, svtkStdString);
  svtkSetMacro(FileName, svtkStdString);
  svtkSetMacro(Width, int);
  svtkSetMacro(Height, int);

  enum RenderType {SPHERES, UGRID, SGRID};
  svtkSetMacro(RenderAs, int);

  //helpers for spheres
  svtkSetMacro(NumberOfPoints, size_t);
  svtkGetMacro(NumberOfPoints, size_t);
  svtkSetMacro(PointPositions, float *);
  std::vector<float> convertedPoints;

  //helpers for ugrid
  std::vector<uint8_t> volumeCellType;
  std::vector<uint32_t> volumeCellIndex;
  std::vector<float> volumeVertexPosition;
  std::vector<float> volumeCellData;
  std::vector<uint32_t> volumeIndex;
  std::vector<float> PointData;
  std::vector<float> TexcoordData;
  // svtkSmartPointer<svtkLookupTable> Lut;
  std::vector<float> Colors;
  std::vector<float> Opacities;
  float TFRange[2] = {0.f, 0.5f};
  OSPRayAnalysisAdaptor::Camera CameraData;
  float ParticleRadius = 0.7f;
  float ParticleColor[3] = {0, 0, 0};
  bool ParticleColorSet = false;
  float BackgroundColor[4] = {0.f, 0.f, 0.f, 1.f};

  svtkSetVector6Macro(Bounds, float);
  void SetCellType(size_t _arg1, uint8_t *_arg2) {
    this->CellTypeSize = _arg1;
    this->CellTypeData = _arg2;
    this->Modified();
  }

  void SetCellIndex(size_t _arg1, uint32_t *_arg2) {
    this->CellIndexSize = _arg1;
    this->CellIndexData = _arg2;
    this->Modified();
  }

  void SetVertexPosition(size_t _arg1, float *_arg2) {
    this->VertexPositionSize = _arg1;
    this->VertexPositionData = _arg2;
    this->Modified();
  }

  void SetCellData(size_t _arg1, float *_arg2) {
    this->CellDataSize = _arg1;
    this->CellDataData = _arg2;
    this->Modified();
  }

  void SetPointData(size_t _arg1, float *_arg2) {
    this->PointDataSize = _arg1;
    this->PointDataData = _arg2;
    this->Modified();
  }

  void SetIndex(size_t _arg1, uint32_t *_arg2) {
    this->IndexSize = _arg1;
    this->IndexData = _arg2;
    this->Modified();
  }

  //helpers for sgrid
  svtkSetMacro(VolumeArrayLength, size_t);
  svtkSetMacro(VolumeArray, float *);
  std::vector<float> convertedVolumeArray;
  void SetDimensions(int *arg) {
    if (!arg) return;
    sgriddimensions[0] = arg[0];
    sgriddimensions[1] = arg[1];
    sgriddimensions[2] = arg[2];
  }
  void SetOrigin(double *arg) {
    if (!arg) return;
    sgridorigin[0] = static_cast<float>(arg[0]);
    sgridorigin[1] = static_cast<float>(arg[1]);
    sgridorigin[2] = static_cast<float>(arg[2]);
  }
  void SetSpacing(double *arg) {
    sgridspacing[0] = static_cast<float>(arg[0]);
    sgridspacing[1] = static_cast<float>(arg[1]);
    sgridspacing[2] = static_cast<float>(arg[2]);
  }
  void SetBounds(double *arg) {
    if (!arg) return;
    sgridbounds[0] = static_cast<float>(arg[0]);
    sgridbounds[1] = static_cast<float>(arg[1]);
    sgridbounds[2] = static_cast<float>(arg[2]);
    sgridbounds[3] = static_cast<float>(arg[3]);
    sgridbounds[4] = static_cast<float>(arg[4]);
    sgridbounds[5] = static_cast<float>(arg[5]);
  }
  void SetDataRange(double *arg) {
    if (!arg) return;
    sdatarange[0] = arg[0];
    sdatarange[1] = arg[1];
    //todo: adjusting LUT range every timestep can be misleading
    //want some user control to modulate this.
    ospSetVec2f(VolumeTF, "value", TFRange[0], TFRange[1]);
    ospCommit(VolumeTF);
  }

private:
  MPI_Comm Communicator;

  int RenderAs = SPHERES; //sphere, unstructured grid, structured grid : todo mesh, sparticle_volume
  //todo (maybe) add another variable to enable thresholds of surfaces
  //todo add another variable to enable isosurfaces of volumes

  //used when RenderType is spheres
  size_t NumberOfPoints;
  float *PointPositions;
  OSPData GeometrySpherePositionData{nullptr};
  OSPData GeometrySphereTexcoordData{nullptr};

  //used when DataType is mesh

  //used when DataType is unstructured volume
  float Bounds[6];
  size_t CellTypeSize;
  uint8_t *CellTypeData;
  size_t CellIndexSize;
  uint32_t *CellIndexData;
  size_t VertexPositionSize;
  float *VertexPositionData;
  size_t CellDataSize;
  float *CellDataData;
  size_t PointDataSize;
  float *PointDataData;
  size_t IndexSize;
  uint32_t *IndexData;
  OSPData VolumeVertexPositionData{nullptr};
  OSPData VolumeIndexData{nullptr};
  OSPData VolumeCellIndexData{nullptr};
  OSPData VolumeCellDataData{nullptr};
  OSPData VolumeCellTypeData{nullptr};

  //used when DataType is structured volume
  int sgriddimensions[3];
  float sgridbounds[6];
  float sgridorigin[3];
  float sgridspacing[3];
  float sdatarange[2];
  size_t VolumeArrayLength;
  float *VolumeArray;
  OSPData VolumeData{nullptr};

  //one container for geometry to show
  OSPGeometricModel GeometricModel{nullptr};
  OSPGeometry Geometry{nullptr};
  OSPMaterial Material{nullptr};
  OSPTexture Texture{nullptr};

  //one container for volume to show
  OSPVolumetricModel VolumetricModel{nullptr};
  //OSPData ColorTF{nullptr};
  //OSPData OpacityTF{nullptr};
  OSPTransferFunction VolumeTF{nullptr};
  OSPData ColorTFData{nullptr};
  OSPData OpacityTFData{nullptr};
  OSPVolume Volume{nullptr};

  //one light
  OSPLight Light{nullptr};

  //one camera position
  OSPCamera Camera{nullptr};

  //top level ospray constructs to build the scene up
  OSPDevice Device{nullptr};
  OSPGroup Group{nullptr};
  OSPInstance Instance{nullptr}; 
  OSPWorld World{nullptr};
  OSPRenderer Renderer{nullptr};

  //to render images into
  int Width;
  int Height;
  OSPFrameBuffer FrameBuffer{nullptr};
  OSPFuture Future{nullptr};

  //each ospmpi distributed rank get a region (technically regions but we use 1)
  std::vector<float> WorldRegion;
  OSPData WorldRegionData{nullptr};

  //to name output image files with
  int FrameNumber{0};
  std::string Directory;
  std::string FileName;

  OSPTransferFunction CreateTransferFunction(int rank);
  void RenderAsSpheresInit(int rank);
  void RenderAsUGridInit(int rank);
  void RenderAsSGridInit(int rank);
  void RenderAsSpheresExecute(float *lo, float *hi);
  void RenderAsUGridExecute(float *lo, float *hi);
  void RenderAsSGridExecute(float *lo, float *hi);
};

svtkStandardNewMacro(OSPRayAnalysisAdaptor::InternalsType);

void OSPRayAnalysisAdaptor::InternalsType::RenderAsSpheresInit(int rank) {
  GeometrySpherePositionData = nullptr;
  Geometry = ospNewGeometry("sphere");
  ospSetObject(Geometry, "sphere.position", GeometrySpherePositionData);
  ospSetObject(Geometry, "sphere.texcoord", GeometrySphereTexcoordData);
  ospSetFloat(Geometry, "radius", ParticleRadius);
  ospCommit(Geometry);

  Material = ospNewMaterial(nullptr, "obj");
  if (!this->Colors.empty() && !this->Opacities.empty())
  {
    int numColors = this->Colors.size()/3;
    Texture = ospNewTexture("texture2d");
    OSPData shared = ospNewSharedData2D(this->Colors.data(), OSP_VEC3F, numColors, 1);
    ospCommit(shared);
    OSPData data = ospNewData2D(OSP_VEC3F, numColors, 1);
    ospCommit(data);
    ospCopyData2D(shared, data, 0, 0);
    ospCommit(data);
    ospRelease(shared);
    ospSetInt(Texture, "format", OSP_TEXTURE_RGB32F);
    ospSetObject(Texture, "data", data);
    ospCommit(Texture);
    ospRelease(data);
    ospSetObject(Material, "map_kd", Texture);
  } else  {
    ospSetVec3f(Material, "kd", (rank % 4 == 0 ? 1.0f : 0.1f), (rank % 4 == 1 ? 1.0f : 0.1f), (rank % 4 == 2 ? 1.0f : 0.1f));
    if (ParticleColorSet)
      ospSetVec3f(Material, "kd", ParticleColor[0], ParticleColor[1], ParticleColor[2]);
  }
  ospCommit(Material);

  GeometricModel = ospNewGeometricModel(nullptr);
  ospSetObject(GeometricModel, "geometry", Geometry);
  ospSetObject(GeometricModel, "material", Material);
  ospCommit(GeometricModel);

  ospSetObjectAsData(Group, "geometry", OSP_GEOMETRIC_MODEL, GeometricModel);
}

OSPTransferFunction OSPRayAnalysisAdaptor::InternalsType::CreateTransferFunction(int rank) {
  auto VolumeTF = ospNewTransferFunction("piecewiseLinear");
  float colors[3] = {((rank * (rank + rank / 3) + 200 * rank) % 256) / 255.f,
                     ((rank * (rank + rank / 7) + 64 * rank) % 256) / 255.f,
                     ((rank * (rank + rank / 9) + 128 * rank) % 256) / 255.f};
  OSPData tmpC = ospNewSharedData(colors, OSP_VEC3F, 1);
  ospCommit(tmpC);
  OSPData ColorTF = ospNewData(OSP_VEC3F, 1);
  ospCommit(ColorTF);
  ospCopyData(tmpC, ColorTF);
  ospCommit(ColorTF);
  ospRelease(tmpC);
  ospSetObject(VolumeTF, "color", ColorTF);

#define OSZ 11
  float opacities[OSZ] = {0.2f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.2f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.2f};
  OSPData tmpO = ospNewSharedData(opacities, OSP_FLOAT, OSZ);
  ospCommit(tmpO);
  OSPData OpacityTF = ospNewData(OSP_FLOAT, OSZ);
  ospCommit(OpacityTF);
  ospCopyData(tmpO, OpacityTF);
  ospCommit(OpacityTF);
  ospRelease(tmpO);
  ospSetObject(VolumeTF, "opacity", OpacityTF);

  if (!this->Colors.empty() && !this->Opacities.empty())
  {
    size_t numColors = this->Colors.size()/3;
    if (numColors != this->Opacities.size())
      std::cerr << "ERROR: ospray: colors and opacities do not match\n";

    OSPData tmpC = ospNewSharedData(this->Colors.data(), OSP_VEC3F, numColors);
    ospCommit(tmpC);
    ColorTFData = ospNewData(OSP_VEC3F, numColors);
    ospCommit(ColorTFData);
    ospCopyData(tmpC, ColorTFData);
    ospCommit(ColorTFData);
    ospRelease(tmpC);
    ospSetObject(VolumeTF, "color", ColorTFData);

    OSPData tmpO = ospNewSharedData(this->Opacities.data(), OSP_FLOAT, numColors);
    ospCommit(tmpO);
    OpacityTFData = ospNewData(OSP_FLOAT, numColors);
    ospCommit(OpacityTFData);
    ospCopyData(tmpO, OpacityTFData);
    ospCommit(OpacityTFData);
    ospRelease(tmpO);
    ospSetObject(VolumeTF, "opacity", OpacityTFData);
  }

  ospSetVec2f(VolumeTF, "value", TFRange[0], TFRange[1]);
  ospCommit(VolumeTF);

  return VolumeTF;
}

void OSPRayAnalysisAdaptor::InternalsType::RenderAsUGridInit(int rank) {
  VolumeTF = CreateTransferFunction(rank);
}


void OSPRayAnalysisAdaptor::InternalsType::RenderAsSGridInit(int rank) {
  VolumeTF = CreateTransferFunction(rank);
}

char const *OSPRayAnalysisAdaptor::InternalsType::Initialize() {
  int rank, size;
  MPI_Comm_rank(Communicator, &rank);
  MPI_Comm_size(Communicator, &size);

#if 1
  bool OSPRAY_MPI_DISTRIBUTED_GPU = (getenv("OSPRAY_MPI_DISTRIBUTED_GPU") && atoi(getenv("OSPRAY_MPI_DISTRIBUTED_GPU")) != 0);
  if (OSPRAY_MPI_DISTRIBUTED_GPU) {
    ospLoadModule("mpi_distributed_gpu");
  } else {
    ospLoadModule("mpi_distributed_cpu");
  }
#else
  ospLoadModule("mpi");
#endif
  Device = ospNewDevice("mpiDistributed");
  ospDeviceCommit(Device);
  ospSetCurrentDevice(Device);

  Group = ospNewGroup();
  if (RenderAs == SPHERES) {
    RenderAsSpheresInit(rank);
  } else if (RenderAs == UGRID) {
    RenderAsUGridInit(rank);
  } else if (RenderAs == SGRID) {
    RenderAsSGridInit(rank);
  }
  ospCommit(Group);

  Instance = ospNewInstance(nullptr);
  ospSetObject(Instance, "group", Group);
  ospCommit(Instance);

  Light = ospNewLight("ambient");
  ospCommit(Light);

  WorldRegionData = nullptr;

  World = ospNewWorld();
  ospSetObjectAsData(World, "instance", OSP_INSTANCE, Instance);
  ospSetObjectAsData(World, "light", OSP_LIGHT, Light);
  ospSetObject(World, "region", WorldRegionData);
  ospCommit(World);

  Camera = ospNewCamera("perspective");
  ospSetFloat(Camera, "aspect", (float)Width / (float)Height);
  auto& up = this->CameraData.Up;
  auto& dir = this->CameraData.Direction;
  auto& pos = this->CameraData.Position;
  ospSetVec3f(Camera, "position", pos[0], pos[1], pos[2]);
  ospSetVec3f(Camera, "direction", dir[0], dir[1], dir[2]);
  ospSetVec3f(Camera, "up", up[0], up[1], up[2]);
  ospSetFloat(Camera, "fovy", this->CameraData.Fovy);
  ospSetFloat(Camera, "focusDistance", this->CameraData.FocusDistance);
  ospCommit(Camera);

  Renderer = ospNewRenderer("mpiRaycast");
  ospSetInt(Renderer, "pixelSamples", 16);
  ospSetVec4f(Renderer, "backgroundColor", BackgroundColor[0], BackgroundColor[1],
    BackgroundColor[2], BackgroundColor[3]);
  ospCommit(Renderer);

  FrameBuffer = ospNewFrameBuffer(Width, Height, OSP_FB_SRGBA, OSP_FB_COLOR | OSP_FB_ACCUM);

  Future = nullptr;

  return nullptr;
}

void OSPRayAnalysisAdaptor::InternalsType::RenderAsSpheresExecute(float *lo, float *hi) {
  if (NumberOfPoints > 0) {
    lo[0] = hi[0] = PointPositions[0*3+0];
    lo[1] = hi[1] = PointPositions[0*3+1];
    lo[2] = hi[2] = PointPositions[0*3+2];
    for (size_t i=1; i<NumberOfPoints; ++i) {
      if (PointPositions[i*3+0] < lo[0]) lo[0] = PointPositions[i*3+0];
      if (PointPositions[i*3+1] < lo[1]) lo[1] = PointPositions[i*3+1];
      if (PointPositions[i*3+2] < lo[2]) lo[2] = PointPositions[i*3+2];
      if (PointPositions[i*3+0] > hi[0]) hi[0] = PointPositions[i*3+0];
      if (PointPositions[i*3+1] > hi[1]) hi[1] = PointPositions[i*3+1];
      if (PointPositions[i*3+2] > hi[2]) hi[2] = PointPositions[i*3+2];
    }
  }

  ospRelease(GeometrySpherePositionData);
  GeometrySpherePositionData = ospNewSharedData(PointPositions, OSP_VEC3F, NumberOfPoints);
  ospCommit(GeometrySpherePositionData);
  ospSetObject(Geometry, "sphere.position", GeometrySpherePositionData);

  if (!TexcoordData.empty()) {
    ospRelease(GeometrySphereTexcoordData);
    GeometrySphereTexcoordData = ospNewSharedData(TexcoordData.data(), OSP_VEC2F, NumberOfPoints);
    ospCommit(GeometrySphereTexcoordData);
    ospSetObject(Geometry, "sphere.texcoord", GeometrySphereTexcoordData);
  }

  ospCommit(Geometry);
}

void OSPRayAnalysisAdaptor::InternalsType::RenderAsUGridExecute(float *lo, float *hi) {
  ospRelease(VolumeCellTypeData);
  VolumeCellTypeData = ospNewSharedData(
    CellTypeData, OSP_UCHAR, CellTypeSize, 0, 1, 0, 1, 0);
  ospCommit(VolumeCellTypeData);

  ospRelease(VolumeCellIndexData);
  VolumeCellIndexData = ospNewSharedData(
    CellIndexData, OSP_UINT, CellIndexSize, 0, 1, 0, 1, 0);
  ospCommit(VolumeCellIndexData);

  ospRelease(VolumeVertexPositionData);
  VolumeVertexPositionData = ospNewSharedData(
    VertexPositionData, OSP_VEC3F, VertexPositionSize, 0, 1, 0, 1, 0);
  ospCommit(VolumeVertexPositionData);

  ospRelease(VolumeCellDataData);
  VolumeCellDataData = ospNewSharedData(
    CellDataData, OSP_FLOAT, CellDataSize, 0, 1, 0, 1, 0);
  ospCommit(VolumeCellDataData);

  ospRelease(VolumeIndexData);
  VolumeIndexData = ospNewSharedData(
    IndexData, OSP_UINT, IndexSize, 0, 1, 0, 1, 0);
  ospCommit(VolumeIndexData);

  ospRelease(Volume);
  Volume = ospNewVolume("unstructured");
  ospSetObject(Volume, "vertex.position", VolumeVertexPositionData);
  ospSetObject(Volume, "index", VolumeIndexData);
  ospSetBool(Volume, "indexPrefixed", false);
  ospSetObject(Volume, "cell.index", VolumeCellIndexData);
  ospSetObject(Volume, "cell.data", VolumeCellDataData);
  ospSetObject(Volume, "cell.type", VolumeCellTypeData);
  ospSetBool(Volume, "hexIterative", true);
  ospCommit(Volume);

  VolumetricModel = ospNewVolumetricModel(nullptr);
  ospSetObject(VolumetricModel, "volume", Volume);
  ospSetObject(VolumetricModel, "transferFunction", VolumeTF);
  ospCommit(VolumetricModel);

  ospSetObjectAsData(Group, "volume", OSP_VOLUMETRIC_MODEL, VolumetricModel);

  lo[0] = sgridbounds[0];
  lo[1] = sgridbounds[2];
  lo[2] = sgridbounds[4];
  hi[0] = sgridbounds[1];
  hi[1] = sgridbounds[3];
  hi[2] = sgridbounds[5];
}

void OSPRayAnalysisAdaptor::InternalsType::RenderAsSGridExecute(float *lo, float *hi) {
  ospRelease(VolumeData);
  VolumeData = ospNewSharedData(VolumeArray, OSP_FLOAT, sgriddimensions[0]-1, 0, sgriddimensions[1]-1, 0, sgriddimensions[2]-1, 0);
  ospCommit(VolumeData);

  ospRelease(Volume);
  Volume = ospNewVolume("structuredRegular");
  ospSetVec3i(Volume, "dimensions", sgriddimensions[0]-1, sgriddimensions[1]-1, sgriddimensions[2]-1);
  ospSetVec3f(Volume, "gridOrigin", sgridorigin[0], sgridorigin[1], sgridorigin[2]);
  ospSetVec3f(Volume, "gridSpacing", sgridspacing[0], sgridspacing[1], sgridspacing[2]);
  ospSetObject(Volume, "data", VolumeData);
  ospCommit(Volume);

  VolumetricModel = ospNewVolumetricModel(nullptr);
  ospSetObject(VolumetricModel, "volume", Volume);
  ospSetObject(VolumetricModel, "transferFunction", VolumeTF);
  ospCommit(VolumetricModel);

  ospSetObjectAsData(Group, "volume", OSP_VOLUMETRIC_MODEL, VolumetricModel);

  lo[0] = sgridbounds[0];
  lo[1] = sgridbounds[2];
  lo[2] = sgridbounds[4];
  hi[0] = sgridbounds[1];
  hi[1] = sgridbounds[3];
  hi[2] = sgridbounds[5];
}

char const *OSPRayAnalysisAdaptor::InternalsType::Execute() {
  int rank, size;
  MPI_Comm_rank(Communicator, &rank);
  MPI_Comm_size(Communicator, &size);

  float lo[3] = {0.f,0.f,0.f};
  float hi[3] = {-1.f,-1.f,-1.f};

  if (RenderAs == SPHERES) {
    RenderAsSpheresExecute(lo, hi);
  } else if (RenderAs == UGRID) {
    RenderAsUGridExecute(lo, hi);
  } else if (RenderAs == SGRID) {
    RenderAsSGridExecute(lo, hi);
  }
  ospCommit(Group);

  ospCommit(Instance);

  float nudge = 1.0f;
  if (RenderAs == SPHERES) {
    nudge = -ParticleRadius;
  }

  WorldRegion.clear();
  WorldRegionData = nullptr;
  if (lo[0] < hi[0] && lo[1] < hi[1] && lo[2] < hi[2]) {
    WorldRegion.insert(WorldRegion.end(), {
      lo[0]+nudge, lo[1]+nudge, lo[2]+nudge,
      hi[0]-nudge, hi[1]-nudge, hi[2]-nudge,
    });
  }
  WorldRegionData = ospNewSharedData(WorldRegion.data(), OSP_BOX3F, WorldRegion.size() / 6);
  ospCommit(WorldRegionData);
  ospSetObject(World, "region", WorldRegionData);
  ospCommit(World);

  ospResetAccumulation(FrameBuffer);
  Future = ospRenderFrame(FrameBuffer, Renderer, Camera, World);
  ospWait(Future, OSP_TASK_FINISHED);
  ospRelease(Future);
  Future = nullptr;

  if (rank == 0) {
    std::string filename = this->Directory + std::string("/") + this->FileName + std::to_string(FrameNumber) + std::string(".ppm");
    const void *fb = ospMapFrameBuffer(FrameBuffer, OSP_FB_COLOR);
    writePPM(filename.c_str(), Width, Height, static_cast<const uint32_t *>(fb));
    ospUnmapFrameBuffer(fb, FrameBuffer);
  }

  ++FrameNumber;

  // cleanup
  ospRelease(VolumetricModel);
  VolumetricModel = nullptr;
  ospRelease(Volume);
  Volume = nullptr;
  ospRelease(VolumeData);
  VolumeData = nullptr;

  return nullptr;
}

char const *OSPRayAnalysisAdaptor::InternalsType::Finalize() {
  ospRelease(FrameBuffer);
  FrameBuffer = nullptr;

  ospRelease(Renderer);
  Renderer = nullptr;

  ospRelease(Camera);
  Camera = nullptr;

  ospRelease(World);
  World = nullptr;

  ospRelease(Light);
  Light = nullptr;

  ospRelease(Instance);
  Instance = nullptr;

  ospRelease(Group);
  Group = nullptr;

  ospRelease(GeometricModel);
  GeometricModel = nullptr;

  ospRelease(Material);
  Material = nullptr;

  ospRelease(Geometry);
  Geometry = nullptr;

  ospRelease(OpacityTFData);
  OpacityTFData = nullptr;

  ospRelease(ColorTFData);
  ColorTFData = nullptr;

  ospRelease(VolumeTF);
  VolumeTF = nullptr;

  if (GeometrySpherePositionData) {
    ospRelease(GeometrySpherePositionData);
    GeometrySpherePositionData = nullptr;
  }

  ospDeviceRelease(Device);
  ospShutdown();

  return nullptr;
}


OSPRayAnalysisAdaptor *OSPRayAnalysisAdaptor::New() {
  auto result = new OSPRayAnalysisAdaptor;
  result->InitializeObjectBase();
  return result;
}

void OSPRayAnalysisAdaptor::SetRenderAs(const char *AsString) {
  if (std::string(AsString) == "SPHERES")
    this->RenderAs = OSPRayAnalysisAdaptor::InternalsType::RenderType::SPHERES;
  if (std::string(AsString) == "UGRID")
    this->RenderAs = OSPRayAnalysisAdaptor::InternalsType::RenderType::UGRID;
  if (std::string(AsString) == "SGRID")
    this->RenderAs = OSPRayAnalysisAdaptor::InternalsType::RenderType::SGRID;
}

int OSPRayAnalysisAdaptor::Initialize() {
  this->Internals = InternalsType::New();
  this->Internals->Colors = this->Colors;
  this->Internals->Opacities = this->Opacities;
  this->Internals->CameraData = this->CameraData;
  this->Internals->ParticleRadius = this->ParticleRadius;
  this->Internals->ParticleColor[0] = this->ParticleColor[0];
  this->Internals->ParticleColor[1] = this->ParticleColor[1];
  this->Internals->ParticleColor[2] = this->ParticleColor[2];
  this->Internals->ParticleColorSet = this->ParticleColorSet;
  this->Internals->BackgroundColor[0] = this->BackgroundColor[0];
  this->Internals->BackgroundColor[1] = this->BackgroundColor[1];
  this->Internals->BackgroundColor[2] = this->BackgroundColor[2];
  this->Internals->BackgroundColor[3] = this->BackgroundColor[3];
  this->Internals->TFRange[0] = this->TFRange[0];
  this->Internals->TFRange[1] = this->TFRange[1];
  this->Internals->SetCommunicator(this->GetCommunicator());
  this->Internals->SetRenderAs(this->RenderAs);
  this->Internals->SetWidth(this->Width);
  this->Internals->SetHeight(this->Height);
  this->Internals->SetDirectory(this->Directory);
  this->Internals->SetFileName(this->FileName);

  char const *error = this->Internals->Initialize();
  if (error != nullptr) {
    std::fprintf(stderr, "Error: %s\n", error);
    return 1;
  }

  return 0;
}

void OSPRayAnalysisAdaptor::RenderAsSpheresImpl(svtkDataObject *dataObject, int rank) {
  svtkPointSet *pointSet;
  pointSet = dynamic_cast<svtkPointSet *>(dataObject);
  if (!pointSet) {
    SENSEI_ERROR("Expected mesh '" << MeshName << "' block " << rank << " to be a vtkPointSet but got " << dataObject->GetClassName())
    return;
  }

  svtkPoints *points;
  points = pointSet->GetPoints();
  if (!points) {
    SENSEI_ERROR("Expected points from mesh '" << MeshName << "' block " << rank << " to exist")
    return;
  }

  svtkDataArray *dataArray;
  dataArray = points->GetData();
  if (!dataArray) {
    SENSEI_ERROR("Expected points from mesh '" << MeshName << "' block " << rank << " to have dataAdaptor")
    return;
  }

  svtkFloatArray *floatArray = svtkFloatArray::SafeDownCast(dataArray);
  svtkDoubleArray *doubleArray = svtkDoubleArray::SafeDownCast(dataArray);
  size_t nComponents = dataArray->GetNumberOfComponents();
  if (nComponents != 3) {
    SENSEI_ERROR("Expected array '" << ArrayName << "' from mesh '" << MeshName << "' block " << rank << " to be a vtkFloatArray with 3 components")
    return;
  }
  if (!dataArray->HasStandardMemoryLayout()) {
    SENSEI_ERROR("Expected array '" << ArrayName << "' from mesh '" << MeshName << "' block " << rank << " to be a vtkFloatArray with 3 components and standard memory layout")
    return;
  }

  size_t nPoints = dataArray->GetNumberOfTuples();
  float *positions;
  if (floatArray)
    positions = floatArray->GetPointer(0);
  else {
    this->Internals->convertedPoints.clear();
    this->Internals->convertedPoints.reserve(nPoints*3);
    double *inPoint = doubleArray->GetPointer(0);
    float *outPoint = this->Internals->convertedPoints.data();
    for (size_t i = 0; i < nPoints; ++i) {
      *outPoint++ = static_cast<float>(*(inPoint++));
      *outPoint++ = static_cast<float>(*(inPoint++));
      *outPoint++ = static_cast<float>(*(inPoint++));
    }
    positions = this->Internals->convertedPoints.data();
  }

  this->Internals->SetNumberOfPoints(nPoints);
  this->Internals->SetPointPositions(positions);

  if (ArrayName != "")
  {
    auto array1 = pointSet->GetPointData()->GetArray(ArrayName.c_str());
    if (array1 == nullptr) {
      SENSEI_ERROR("Expected CellData->GetScalars() from ugrid from mesh '" << MeshName << "' to exist")
      return;
    }
    this->Internals->PointData.resize(array1->GetNumberOfValues(), 0.0f);
    for (size_t i=0; i<array1->GetNumberOfValues(); ++i) {
      this->Internals->PointData[i] = static_cast<float>(array1->GetTuple1(i));
    }
    this->Internals->SetPointData(
      array1->GetNumberOfTuples(), this->Internals->PointData.data());

    double r[2];
    array1->GetRange(r);
    double global;
    MPI_Allreduce(&r[0], &global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    r[0] = global;
    MPI_Allreduce(&r[1], &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    r[1] = global;
    this->Internals->SetDataRange(r);

    this->Internals->TexcoordData.resize(this->Internals->GetNumberOfPoints() * 2);
    for(int i = 0; i < this->Internals->GetNumberOfPoints(); i++) {
      const float p = this->Internals->PointData[i];
      this->Internals->TexcoordData[i * 2] = (p-r[0])/(r[1]-r[0]);
      this->Internals->TexcoordData[i * 2 + 1] = 0.f;
    }
  }
}

void OSPRayAnalysisAdaptor::RenderAsUGridImpl(svtkDataObject *dataObject, int rank) {
  using UnstructuredGrid = svtkUnstructuredGrid;
  UnstructuredGrid *unstructuredGrid;
  if ((unstructuredGrid = UnstructuredGrid::SafeDownCast(dataObject)) == nullptr) {
    SENSEI_ERROR("Expected data object from mesh '" << MeshName << "' to be a vtkUnstructuredGrid")
    return;
  }
  unstructuredGrid->Register(this);

#ifdef ENABLE_VTK_FILTERS_PARALLEL_GEOMETRY
  // Optionally redistribute the vtkUnstructuredGrid with D3 filter.
  if (UseD3) {
    using Filter = vtkPDistributedDataFilter;
    vtkNew<Filter> filter;
    filter->GetKdtree()->AssignRegionsRoundRobin();
    vtkUnstructuredGrid *vug = SVTKUtils::VTKObjectFactory::New(unstructuredGrid);
    filter->SetInputData(vug);
    filter->SetBoundaryMode(0);
    filter->SetUseMinimalMemory(1);
    filter->SetMinimumGhostLevel(0);
    filter->RetainKdtreeOn();

    filter->Update();

    vug->Delete();
    vug = vtkUnstructuredGrid::SafeDownCast(filter->GetOutput());
    unstructuredGrid->UnRegister(this);
    unstructuredGrid = SVTKUtils::SVTKObjectFactory::New(vug);
    unstructuredGrid->Register(this);
  }
#endif
  //unstructuredGrid->PrintSelf(std::cerr, svtkIndent(0));

  // Extract data from vtkUnstructuredGrid.

  double bds[6];
  unstructuredGrid->GetBounds(bds);
  this->Internals->SetBounds(bds);

  {
    using Array = svtkUnsignedCharArray;
    Array *array = unstructuredGrid->GetCellTypesArray();
    if (array == nullptr) {
      SENSEI_ERROR("Expected CellTypesArray from ugrid from mesh '" << MeshName << "' to be a vtkUnsignedCharArray")
      return;
    }
    this->Internals->volumeCellType.resize(array->GetNumberOfValues(), 0);
    for (size_t i=0; i<array->GetNumberOfValues(); ++i) {
      this->Internals->volumeCellType[i] = array->GetValue(i);
    }
    this->Internals->SetCellType(
      array->GetNumberOfTuples(), this->Internals->volumeCellType.data());
  }

  {
    using Array = svtkIdTypeArray;
    Array *array = unstructuredGrid->GetCellLocationsArray();
    if (array == nullptr) {
      SENSEI_ERROR("Expected CellLocationsArray from ugrid from mesh '" << MeshName << "' to be a vtkIdTypeArray")
      return;
    }
    this->Internals->volumeCellIndex.resize(array->GetNumberOfValues(), 0);
    for (size_t i=0; i<array->GetNumberOfValues(); ++i) {
      this->Internals->volumeCellIndex[i] = array->GetValue(i);
    }
    this->Internals->SetCellIndex(
      array->GetNumberOfTuples(), this->Internals->volumeCellIndex.data());
  }

  {
    using Array = svtkDoubleArray;
    Array *array = Array::SafeDownCast(unstructuredGrid->GetPoints()->GetData());
    if (array == nullptr) {
      SENSEI_ERROR("Expected GetPoints->GetData() from ugrid from mesh '" << MeshName << "' to be a vtkDoubleArray")
      return;
    }
    this->Internals->volumeVertexPosition.resize(array->GetNumberOfValues(), 0.0f);
    for (size_t i=0; i<array->GetNumberOfValues(); ++i) {
      this->Internals->volumeVertexPosition[i] = array->GetValue(i);
    }
    this->Internals->SetVertexPosition(
      array->GetNumberOfTuples(), this->Internals->volumeVertexPosition.data());
  }

  {
    auto array1 = unstructuredGrid->GetCellData()->GetArray(ArrayName.c_str());
    if (array1 == nullptr) {
      SENSEI_ERROR("Expected CellData->GetScalars() from ugrid from mesh '" << MeshName << "' to exist")
      return;
    }
    this->Internals->volumeCellData.resize(array1->GetNumberOfValues(), 0.0f);
    for (size_t i=0; i<array1->GetNumberOfValues(); ++i) {
      this->Internals->volumeCellData[i] = static_cast<float>(array1->GetTuple1(i));
    }
    this->Internals->SetCellData(
      array1->GetNumberOfTuples(), this->Internals->volumeCellData.data());

    double r[2];
    array1->GetRange(r);
    double global;
    MPI_Allreduce(&r[0], &global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    r[0] = global;
    MPI_Allreduce(&r[1], &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    r[1] = global;
    this->Internals->SetDataRange(r);
  }

  {
    using Array = svtkTypeInt64Array;
    Array *array = Array::SafeDownCast(unstructuredGrid->GetCells()->GetConnectivityArray());
    if (array == nullptr) {
      SENSEI_ERROR("Expected Cells->GetConnectivityArray() from ugrid from mesh '" << MeshName << "' to be a vtkTypeInt64Array")
      return;
    }
    this->Internals->volumeIndex.resize(array->GetNumberOfValues(), 0.0f);
    for (size_t i=0; i<array->GetNumberOfValues(); ++i) {
      this->Internals->volumeIndex[i] = array->GetValue(i);
    }
    this->Internals->SetIndex(
      array->GetNumberOfTuples(), this->Internals->volumeIndex.data());
  }
  unstructuredGrid->Delete();
}

void OSPRayAnalysisAdaptor::RenderAsSGridImpl(svtkDataObject *dataObject, int rank) {
  svtkImageData *imageData;
  imageData = dynamic_cast<svtkImageData *>(dataObject);
  if (!imageData) {
    SENSEI_ERROR("Expected mesh '" << MeshName << "' block " << rank << " to be a vtkImageData but got " << dataObject->GetClassName())
    return;
  }

  //imageData->PrintSelf(std::cerr, svtkIndent(0));
  svtkDataArray *dataArray = nullptr;
  if (this->Association == "point")
    dataArray = imageData->GetPointData()->GetArray(this->ArrayName.c_str());
  else if (this->Association == "cell")
    dataArray = imageData->GetCellData()->GetArray(this->ArrayName.c_str());
  if (!dataArray) {
    SENSEI_ERROR("Expected points from mesh '" << MeshName << "' block " << rank << " to have dataAdaptor")
    return;
  }
  if (svtkFloatArray::SafeDownCast(dataArray) == nullptr) {
    SENSEI_ERROR("Expected points array from mesh '" << MeshName << "' block " << rank << " to contain floats")
    return;
  }
  this->Internals->SetVolumeArrayLength(dataArray->GetNumberOfTuples());
  double r[2];
  dataArray->GetRange(r);
  double global;
  MPI_Allreduce(&r[0], &global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  r[0] = global;
  MPI_Allreduce(&r[1], &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  r[1] = global;
  this->Internals->SetDataRange(r);
  this->Internals->SetVolumeArray((float*)dataArray->GetVoidPointer(0));

  double origin[3];
  double spacing[3];
  int dimensions[3];
  imageData->GetOrigin(origin);
  imageData->GetSpacing(spacing);
  imageData->GetDimensions(dimensions);
  double* bds = imageData->GetBounds();
  origin[0] = bds[0];
  origin[1] = bds[2];
  origin[2] = bds[4];

  this->Internals->SetDimensions(dimensions);
  this->Internals->SetOrigin(origin);
  this->Internals->SetSpacing(spacing);
  this->Internals->SetBounds(bds);

}

bool OSPRayAnalysisAdaptor::Execute(
        sensei::DataAdaptor *dataAdaptor,
        sensei::DataAdaptor **outDataAdaptor) {
  *outDataAdaptor = nullptr; //out output is rendered files so we don't pass output downstream
  int rank, size;
  MPI_Comm_rank(GetCommunicator(), &rank);
  MPI_Comm_size(GetCommunicator(), &size);

  svtkDataObject *mesh;
  if (dataAdaptor->GetMesh(MeshName.c_str(), /*structureOnly=*/false, mesh)) {
    SENSEI_ERROR("Failed to get mesh '" << MeshName << "'")
    return false;
  }

  if (ArrayName != "") {
    auto association = svtkDataObject::POINT;
    if (Association == "cell") association = svtkDataObject::CELL;
    if (dataAdaptor->AddArray(mesh, MeshName.c_str(), association, ArrayName.c_str())) {
      SENSEI_ERROR("Failed to get array '" << ArrayName << "' from mesh '" << MeshName << "' block " << rank)
      return false;
    }
  }

  svtkMultiBlockDataSet *multiBlockDataSet;
  multiBlockDataSet = dynamic_cast<svtkMultiBlockDataSet *>(mesh);
  if (!multiBlockDataSet) {
    SENSEI_ERROR("Expected mesh '" << MeshName << "' to be a vtkMultiBlockDataSet")
    return false;
  }

  if (static_cast<int>(multiBlockDataSet->GetNumberOfBlocks()) != size) {
    SENSEI_ERROR("Expected mesh '" << MeshName << "' to have only " << size << " blocks")
    return false;
  }

  svtkDataObject *dataObject;
  dataObject = multiBlockDataSet->GetBlock(rank);
  if (dataObject == nullptr) {
    SENSEI_ERROR("Expected mesh '" << MeshName << "' block " << rank << " to be non-null")
    return false;
  }

  if (this->RenderAs == OSPRayAnalysisAdaptor::InternalsType::SPHERES) {
    this->RenderAsSpheresImpl(dataObject, rank);
  } else if (this->RenderAs == OSPRayAnalysisAdaptor::InternalsType::UGRID) {
    this->RenderAsUGridImpl(dataObject, rank);
  } else if (this->RenderAs == OSPRayAnalysisAdaptor::InternalsType::SGRID) {
    this->RenderAsSGridImpl(dataObject, rank);
  }

  this->Internals->SetCommunicator(GetCommunicator());

  char const *error = this->Internals->Execute();
  if (error != nullptr) {
    std::fprintf(stderr, "Error: %s\n", error);
    return false;
  }

  mesh->Delete();

  return true;
}

int OSPRayAnalysisAdaptor::Finalize() {
  char const *error = this->Internals->Finalize();
  if (error != nullptr) {
    std::fprintf(stderr, "Error: %s\n", error);
    return 1;
  }

  this->Internals->Delete();

  return 0;
}

} /* namespace sensei */
