#include "CinemaHelper.h"
#include "Profiler.h"

#include <vector>
#include <map>
#include <math.h>
#include <sstream>
#include <algorithm>

#include <vtksys/SystemTools.hxx>

#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#if defined(ENABLE_CATALYST) || defined (ENABLE_VTK_MPI)
#  include <vtkMultiProcessController.h>
#endif
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#ifdef ENABLE_CATALYST
#  include <vtkSMPropertyHelper.h>
#  include <vtkSMRenderViewProxy.h>
#  include <vtkSMRepresentationProxy.h>
#  include <vtkSMViewProxy.h>
#endif
#include <vtkUnsignedCharArray.h>

#if defined(ENABLE_CATALYST) || defined (ENABLE_VTK_RENDERING)
#  include <vtkCamera.h>
#  include <vtkActor.h>
#  include <vtkRenderWindow.h>
#  include <vtkRenderer.h>
#  include <vtkWindowToImageFilter.h>
#endif
#ifdef ENABLE_CATALYST
#  include <vtkIceTCompositePass.h>
#  include <vtkCameraPass.h>
#  include <vtkLightingMapPass.h>
#endif


#if !defined(_WIN32) || defined(__CYGWIN__)
# include <unistd.h> /* unlink */
#else
# include <io.h> /* unlink */
#endif

namespace sensei
{
// --------------------------------------------------------------------------
// Helper functions
// --------------------------------------------------------------------------

double* normalize(double* vect, double tolerance=0.00001)
{
  double mag2 = (vect[0] * vect[0]) + (vect[1] * vect[1]) + (vect[2] * vect[2]);
  if (std::abs(mag2 - 1.0) > tolerance)
    {
    double mag = sqrt(mag2);
    vect[0] /= mag;
    vect[1] /= mag;
    vect[2] /= mag;
    }

  return vect;
}

// --------------------------------------------------------------------------

double* q_mult(double* q1, double* q2, double* out)
{
  double w1 = q1[0];
  double x1 = q1[1];
  double y1 = q1[2];
  double z1 = q1[3];

  double w2 = q2[0];
  double x2 = q2[1];
  double y2 = q2[2];
  double z2 = q2[3];

  out[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
  out[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
  out[2] = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2;
  out[3] = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2;

  return out;
}

// --------------------------------------------------------------------------

double* q_conjugate(double* q, double* out)
{
  out[0] = q[0];
  out[1] = -q[1];
  out[2] = -q[2];
  out[3] = -q[3];
  return out;
}

// --------------------------------------------------------------------------

double* qv_mult(double* q1, double* v1, double* out)
{
  double out0[4];
  double out1[4];
  double out2[4];

  double q2[4];
  q2[0] = 0.0;
  q2[1] = v1[0];
  q2[2] = v1[1];
  q2[3] = v1[2];

  q_mult(q_mult(q1, q2, out0), q_conjugate(q1, out1), out2);
  out[0] = out2[1];
  out[1] = out2[2];
  out[2] = out2[3];

  return out;
}

// --------------------------------------------------------------------------

double* axisangle_to_q(double* v, double theta, double* out)
{
  normalize(v);
  double halfTheta = theta / 2;
  out[0] = cos(halfTheta);
  out[1] = v[0] * sin(halfTheta);
  out[2] = v[1] * sin(halfTheta);
  out[3] = v[2] * sin(halfTheta);
  return out;
}

// --------------------------------------------------------------------------

double* vectProduct(double* axisA, double* axisB, double* out)
{
  out[0] = axisA[1] * axisB[2] - axisA[2] * axisB[1]; // ya*zb - za*yb
  out[1] = axisA[2] * axisB[0] - axisA[0] * axisB[2]; // za*xb - xa*zb
  out[2] = axisA[0] * axisB[1] - axisA[1] * axisB[0]; // xa*yb - ya*xb

  normalize(out);
  return out;
}

// --------------------------------------------------------------------------
double dotProduct(double* vecA, double* vecB)
{
  return (vecA[0] * vecB[0]) + (vecA[1] * vecB[1]) + (vecA[2] * vecB[2]);
}

// --------------------------------------------------------------------------
double* rotate(double* axis, double angle, double* center, double* point, double* out)
{
  double angleInRad = 3.141592654 * angle / 180.0;

  double rotation[4];
  axisangle_to_q(axis, angleInRad, rotation);

  double tPoint[3];
  tPoint[0] = (point[0] - center[0]);
  tPoint[1] = (point[1] - center[1]);
  tPoint[2] = (point[2] - center[2]);

  qv_mult(rotation, tPoint, out);

  out[0] += center[0];
  out[1] += center[1];
  out[2] += center[2];

  return out;
}

// --------------------------------------------------------------------------

std::vector<std::string>& split(const std::string &str, const std::string &delimiter, std::vector<std::string> &output)
{
    size_t start = 0;
    size_t end = 0;
    while (end != std::string::npos)
    {
      end = str.find(delimiter, start);
      output.push_back(str.substr(start, (end == std::string::npos) ? std::string::npos : end - start));
      start = (end > (std::string::npos - delimiter.size()))
        ? std::string::npos
        : end + delimiter.size();
    }

    return output;
}

// --------------------------------------------------------------------------

void convertToDoubles(std::vector<std::string> &input, double* output)
{
  int size = input.size();
  for(int i = 0; i < size; i++)
    {
    output[i] = std::stod(input[i]);
    }
}

// --------------------------------------------------------------------------
// Helper Struct for pixel sorting
// --------------------------------------------------------------------------

struct Pixel
{
  unsigned char Index;
  float Depth;

  Pixel() : Index(0), Depth(0.0) {}
  ~Pixel() {}
  Pixel(const Pixel& other)
  {
    this->Index = other.Index;
    this->Depth = other.Depth;
  }
  void operator=(const Pixel& other)
  {
    this->Index = other.Index;
    this->Depth = other.Depth;
  }
  // void operator<(const Pixel& other)
  // {
  //   return this->Depth < other.Depth;
  // }
};

bool pixelComp(Pixel& a, Pixel& b)
{
  return a.Depth < b.Depth;
}

// --------------------------------------------------------------------------
// Internals
// --------------------------------------------------------------------------

struct CinemaHelper::Internals
{
  double ImageSize[2];
  int NumberOfTimeSteps;
  int SampleSize;
  std::string WorkingDirectory;
  int NumberOfCameraPositions;
  double* CameraPositions; // (focalPoint[3], position[3], viewUp[3]) * NumberOfCameraPositions
  int NumberOfCameraArgs;
  double* CameraArgs;
  bool IsRoot;
  int PID;
  int CurrentCameraPosition;
#ifdef ENABLE_CATALYST
  std::vector<vtkSmartPointer<vtkSMRepresentationProxy>> Representations;
#endif
  std::vector<vtkSmartPointer<vtkActor>> Actors;
  std::string CaptureMethod;
  // Scalars
  std::string ScalarName;
  std::string ScalarAssociation;
  // Contours
  std::vector<double> Contours;
  // JSON metadata handling
  std::string LAYER_CODES;
  std::string JSONTypes;
  std::string JSONCameraArgsOrder;
  std::string JSONCameraArgs;
  std::string JSONCameraPattern;
  std::map<std::string, std::string> JSONData;
  std::string JSONExtraMetadata;
  std::vector<std::string> JSONPipeline;
  std::vector<std::string> JSONCompositePipeline;


  Internals() : NumberOfTimeSteps(0), NumberOfCameraPositions(0), CameraPositions(nullptr), NumberOfCameraArgs(0), CameraArgs(nullptr), CurrentCameraPosition(0), LAYER_CODES("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
  {
    this->ImageSize[0] = this->ImageSize[1] = 512;
    this->SampleSize = 1024;
#if defined(ENABLE_CATALYST) || defined (ENABLE_VTK_MPI)
    vtkMultiProcessController* controller = vtkMultiProcessController::GetGlobalController();
    this->IsRoot = (controller->GetLocalProcessId() == 0);
    this->PID = controller->GetLocalProcessId();
#else
    this->IsRoot = true;
    this->PID = 0;
#endif
  }

  ~Internals()
  {
    if (this->CameraPositions)
    {
      delete[] this->CameraPositions;
    }
    if (this->CameraArgs)
    {
      delete[] this->CameraArgs;
    }
  }

  std::string getDataAbsoluteFilePath(const std::string &fileName, bool createDirectory)
  {
    std::ostringstream resultPath;
    // Basepath
    resultPath << this->WorkingDirectory << "/";
    // Time
    resultPath << this->NumberOfTimeSteps << "/";
    // Camera
    for (int cameraArgIdx = 0; cameraArgIdx < this->NumberOfCameraArgs; cameraArgIdx++)
    {
      if (cameraArgIdx > 0)
      {
        resultPath << "_";
      }
      resultPath << this->CameraArgs[this->CurrentCameraPosition * this->NumberOfCameraArgs + cameraArgIdx];
    }

    if (createDirectory)
    {
      vtksys::SystemTools::MakeDirectory(resultPath.str().c_str());
    }

    resultPath << "/";
    // Filename
    resultPath << fileName;

    return resultPath.str();
  }

  void UpdateSphericalCameraPosition(double* initialFocalPoint, double* initialPosition, double* initialViewUp, int numberOfPhi, double* phiAngles, int numberOfTheta, double* thetaAngles)
  {
    // Reserve args for Phi and Theta
    this->NumberOfCameraArgs = 2;

    if (this->CameraPositions)
    {
      delete[] this->CameraPositions;
    }
    if (this->CameraArgs)
    {
      delete[] this->CameraArgs;
    }

    this->NumberOfCameraPositions = numberOfPhi * numberOfTheta;
    this->CameraPositions = new double[this->NumberOfCameraPositions * 9];
    this->CameraArgs = new double[this->NumberOfCameraPositions * this->NumberOfCameraArgs];

    double center[3];
    center[0] = center[1] = center[2] = 0;
    int cameraPosition = 0;

    for(int thetaIdx = 0; thetaIdx < numberOfTheta; thetaIdx++)
      {
      double theta = thetaAngles[thetaIdx];
      for(int phiIdx = 0; phiIdx < numberOfPhi; phiIdx++)
        {
          double* cameraData = &this->CameraPositions[cameraPosition * 3 * 3]; // focalPoint[3], position[3], viewUp[3]
          double* cameraArgs = &this->CameraArgs[cameraPosition * this->NumberOfCameraArgs];
          double phi = phiAngles[phiIdx];

          // Configure Camera args
          cameraArgs[0] = theta;
          cameraArgs[1] = phi;

          double phiPos[3];
          rotate(initialViewUp, -phi, initialFocalPoint, initialPosition, phiPos);

          double cameraDir[3];
          cameraDir[0] = initialFocalPoint[0] - phiPos[0];
          cameraDir[1] = initialFocalPoint[1] - phiPos[1];
          cameraDir[2] = initialFocalPoint[2] - phiPos[2];

          double thetaAxis[3];
          vectProduct(initialViewUp, cameraDir, thetaAxis);

          double* thetaPhiPos = &cameraData[3]; // Position
          rotate(thetaAxis, theta, initialFocalPoint, phiPos, thetaPhiPos);

          double* viewUp = &cameraData[6];
          rotate(thetaAxis, theta, center, initialViewUp, viewUp);

          // Fix focal point
          cameraData[0] = initialFocalPoint[0];
          cameraData[1] = initialFocalPoint[1];
          cameraData[2] = initialFocalPoint[2];

          // Shift to next camera data position
          cameraPosition++;
        }
      }

    // Build JSON structure
    std::ostringstream jsonArgsSection;
    jsonArgsSection
      << "\"theta\": {"                                            << endl
      << "   \"default\": " << int((numberOfTheta - 1) / 2) << "," << endl
      << "   \"ui\": \"slider\","                                  << endl
      << "   \"values\": ["                                        << endl;

    for(int thetaIdx = 0; thetaIdx < numberOfTheta; thetaIdx++)
      {
      if (thetaIdx > 0)
        {
        jsonArgsSection << ", ";
        }
      jsonArgsSection << "\"" << thetaAngles[thetaIdx] << "\"";
      }

    jsonArgsSection
      << "\n   ],"                       << endl
      << "   \"bind\": {"                << endl
      << "     \"mouse\": {"             << endl
      << "         \"drag\": {"          << endl
      << "           \"modifier\": 0,"   << endl
      << "           \"coordinate\": 1," << endl
      << "           \"step\": 30,"      << endl
      << "           \"orientation\": 1" << endl
      << "         }"                    << endl
      << "       }"                      << endl
      << "   }"                          << endl
      << "},"                            << endl;

    jsonArgsSection
      << "\"phi\": {"               << endl
      << "   \"default\": 0,"       << endl
      << "   \"ui\": \"slider\","   << endl
      << "   \"loop\": \"modulo\"," << endl
      << "   \"values\": ["         << endl;

    for(int phiIdx = 0; phiIdx < numberOfPhi; phiIdx++)
      {
      if (phiIdx > 0)
        {
        jsonArgsSection << ", ";
        }
      jsonArgsSection << "\"" << phiAngles[phiIdx] << "\"";
      }

    jsonArgsSection
      << "\n   ],"                       << endl
      << "   \"bind\": {"                << endl
      << "     \"mouse\": {"             << endl
      << "         \"drag\": {"          << endl
      << "           \"modifier\": 0,"   << endl
      << "           \"coordinate\": 0," << endl
      << "           \"step\": 10,"      << endl
      << "           \"orientation\": 1" << endl
      << "         }"                    << endl
      << "       }"                      << endl
      << "   }"                          << endl
      << "},";

    this->JSONCameraArgs = jsonArgsSection.str();
    this->JSONCameraArgsOrder = "\"theta\", \"phi\",";
    this->JSONCameraPattern = "{theta}_{phi}";
  }
};

// --------------------------------------------------------------------------
// Helper class
// --------------------------------------------------------------------------

CinemaHelper::CinemaHelper()
{
  this->Data = new Internals();
}

// --------------------------------------------------------------------------
CinemaHelper::~CinemaHelper()
{
  delete this->Data;
}

// --------------------------------------------------------------------------
void CinemaHelper::SetImageSize(int width, int height)
{
  this->Data->ImageSize[0] = width;
  this->Data->ImageSize[1] = height;
}

// --------------------------------------------------------------------------
void CinemaHelper::SetWorkingDirectory(const std::string& path)
{
  this->Data->WorkingDirectory = path;
}

// --------------------------------------------------------------------------
void CinemaHelper::SetSampleSize(int ssz)
{
  this->Data->SampleSize = ssz;
}

// --------------------------------------------------------------------------
void CinemaHelper::AddTimeEntry()
{
  this->Data->NumberOfTimeSteps++;
}

// --------------------------------------------------------------------------
void CinemaHelper::WriteMetadata()
{
  if (!this->Data->IsRoot)
    {
    return;
    }

  std::ostringstream filePath;
  filePath << this->Data->WorkingDirectory << "/index.json";
  std::ofstream fp(filePath.str().c_str(), ios::out);

  if (fp.fail())
    {
    std::cout << "Unable to open file: "<< filePath.str().c_str() << std::endl;
    }
  else
    {
    fp
      << "{"                                << endl
      << "  \"metadata\": {"                << endl
      << "    \"backgroundColor\": \"#ffffff\"" << endl;
    if (!this->Data->JSONExtraMetadata.empty())
    {
      fp << this->Data->JSONExtraMetadata << endl;
    }
    fp
      << "  },"                             << endl
      << "  \"type\": ["                    << endl
      << "       " << this->Data->JSONTypes << endl
      << "   ],"                            << endl
      << "  \"arguments_order\": ["         << endl
      ;
    if (this->Data->NumberOfCameraArgs > 0)
    {
      fp << this->Data->JSONCameraArgsOrder    << endl;
    }
    fp
      << "     \"time\""                    << endl
      << "  ],"                             << endl
      << "  \"arguments\": {"               << endl
      ;
    if (this->Data->NumberOfCameraArgs > 0)
    {
      fp << this->Data->JSONCameraArgs      << endl;
    }
    fp
      << "     \"time\": {"                 << endl
      << "        \"loop\": \"modulo\","    << endl
      << "        \"ui\": \"slider\","      << endl
      << "        \"values\": ["            << endl;

    // > Add time entries =======================
    for (int tIdx = 1; tIdx <= this->Data->NumberOfTimeSteps; tIdx++)
      {
      if (tIdx > 1)
        {
        fp << ", ";
        }
      fp << "\"" << tIdx << "\"";
      }
    // < Add time entries =======================

    fp
      << "\n        ]"                      << endl
      << "     }"                           << endl
      << "  },"                             << endl
      << "  \"data\": ["                    << endl;

    // > Add data entries =======================
    std::map<std::string,std::string>::iterator it;
    for (it=this->Data->JSONData.begin(); it!=this->Data->JSONData.end(); ++it)
      {
      if (it != this->Data->JSONData.begin())
        {
        fp << "," << endl;
        }
      fp << it->second;
      }
    // < Add time entries =======================

    fp << "\n  ]";

    // > Handle pipeline =======================
    if (this->Data->JSONPipeline.size())
      {
      fp
        << ","                                                            << endl
        << "   \"SortedComposite\": {"                                    << endl
        << "     \"ranges\": { "                                          << endl
        << "       \"scalar\": [0, 1]"                                    << endl // FIXME
        << "     },"                                                      << endl
        << "     \"layers\": " << this->Data->JSONPipeline.size() << ","  << endl
        << "     \"pipeline\": ["                                         << endl;

      // > Add pipeline entries =======================
      std::vector<std::string>::iterator itv;
      for (itv = this->Data->JSONPipeline.begin(); itv != this->Data->JSONPipeline.end(); ++itv)
        {
        if (itv != this->Data->JSONPipeline.begin())
          {
          fp << "," << endl;
          }
        fp << *itv;
        }
      // < Add pipeline entries =======================

      fp
        << "     ]," << endl
        << "     \"dimensions\": ["
        << this->Data->ImageSize[0] << ", " << this->Data->ImageSize[1]
        << "]," << endl
        << "     \"light\": [\"intensity\"]" << endl
        << "   }," << endl
        << "   \"CompositePipeline\": {" << endl
        << "      \"default_pipeline\": \"";

      // > Add default pipeline =======================
      for (size_t i = 0; i < this->Data->JSONCompositePipeline.size(); i++)
        {
        fp << this->Data->LAYER_CODES[i] << "A";
        }
      // < Add default pipeline =======================

      fp
        << "\"," << endl
        << "      \"layers\": [" << endl;

      // > Add layers =======================
      for (size_t i = 0; i < this->Data->JSONCompositePipeline.size(); i++)
        {
        if (i > 0)
          {
          fp << ", ";
          }
        fp << "\"" << this->Data->LAYER_CODES[i] << "\"";
        }
      // < Add layers =======================

      fp
        << "      ]," << endl
        << "      \"pipeline\": [" << endl;
        // => add

      // > Add composite pipeline entries =======================
      for (itv = this->Data->JSONCompositePipeline.begin(); itv != this->Data->JSONCompositePipeline.end(); ++itv)
        {
        if (itv != this->Data->JSONCompositePipeline.begin())
          {
          fp << "," << endl;
          }
        fp << *itv;
        }
      // < Add composite pipeline entries =======================

      fp
        << "      ]," << endl
        << "      \"layer_fields\": {" << endl;
        // => Add layer fields

      // > Add layers fields =======================
      for (size_t i = 0; i < this->Data->JSONCompositePipeline.size(); i++)
        {
        if (i > 0)
          {
          fp << "," << endl;
          }
        fp << "\"" << this->Data->LAYER_CODES[i] << "\": [\"A\"]";
        }
      // < Add layers fields =======================

      fp
        << "\n      }," << endl
        << "      \"fields\": {" << endl
        << "         \"A\": \"scalar\"" << endl // FIXME
        << "      }" << endl
        << "   }";

      }
    // < Add time entries =======================

    fp
      << "\n}" << endl;
    fp.flush();
    // std::cout << "Writting file: "<< filePath.str().c_str() << std::endl;
    }
}

// --------------------------------------------------------------------------
void CinemaHelper::SetCameraConfig(const std::string& config)
{
  std::vector<std::string> args;
  split(config, ":", args);
  if (args[0] == "spherical")
    {
      std::vector<std::string> focalPoint;
      std::vector<std::string> position;
      std::vector<std::string> viewUp;
      std::vector<std::string> phiAnglesString;
      std::vector<std::string> thetaAnglesString;

      split(args[1], ",", focalPoint);
      split(args[2], ",", position);
      split(args[3], ",", viewUp);
      split(args[4], ",", phiAnglesString);
      split(args[5], ",", thetaAnglesString);

      double initialFocalPoint[3];
      double initialPosition[3];
      double initialViewUp[3];
      int numberOfPhi = phiAnglesString.size();
      int numberOfTheta = thetaAnglesString.size();
      double *phiAngles = new double[numberOfPhi];
      double *thetaAngles = new double[numberOfTheta];

      convertToDoubles(focalPoint, initialFocalPoint);
      convertToDoubles(position, initialPosition);
      convertToDoubles(viewUp, initialViewUp);
      convertToDoubles(phiAnglesString, phiAngles);
      convertToDoubles(thetaAnglesString, thetaAngles);

      this->Data->UpdateSphericalCameraPosition(initialFocalPoint, initialPosition, initialViewUp, numberOfPhi, phiAngles, numberOfTheta, thetaAngles);

      delete[] phiAngles;
      delete[] thetaAngles;
    }
  else if (args[0] == "none")
    {
    }
}
// --------------------------------------------------------------------------
void CinemaHelper::SetExportType(const std::string& exportType)
{
  if (exportType == "sorted-composite")
    {
    this->Data->CaptureMethod = "CaptureSortedCompositeData";
    this->Data->JSONTypes = "\"tonic-query-data-model\", \"sorted-composite\", \"multi-color-by\"";
    this->Data->JSONData["order"] = "{ \"pattern\": \"{time}/";
    this->Data->JSONData["order"] += this->Data->JSONCameraPattern;
    this->Data->JSONData["order"] += "/order.uint8\", \"type\": \"array\", \"name\": \"order\" }";
    this->Data->JSONData["intensity"] = "{ \"pattern\": \"{time}/";
    this->Data->JSONData["intensity"] += this->Data->JSONCameraPattern;
    this->Data->JSONData["intensity"] += "/intensity.uint8\", \"type\": \"array\", \"name\": \"intensity\" }";
    }
  else if (exportType == "vtk-volume")
    {
    this->Data->JSONTypes = "\"tonic-query-data-model\", \"vtk-volume\"";
    this->Data->JSONData["scene"] = "{ \"pattern\": \"{time}/volume.json\", \"rootFile\": true, \"name\": \"scene\", \"type\": \"json\" }";
    }
  else if (exportType == "cdf")
    {
    this->Data->JSONTypes = "\"tonic-query-data-model\", \"cdf\"";
    }
  else
    {
    this->Data->CaptureMethod = "CaptureImage";
    this->Data->JSONTypes = "\"tonic-query-data-model\"";
    this->Data->JSONData["image"] = "{ \"pattern\": \"{time}/";
    this->Data->JSONData["image"] += this->Data->JSONCameraPattern;
    this->Data->JSONData["image"] += "/image.jpg\", \"type\": \"blob\", \"name\": \"image\", \"mimeType\": \"image/jpg\" }";
    }
}

// --------------------------------------------------------------------------
void CinemaHelper::SetContours(const std::string& values)
{
  this->Data->Contours.empty();
  std::vector<std::string> valuesAsString;
  split(values, ",", valuesAsString);
  std::vector<std::string>::iterator iter;
  for (iter = valuesAsString.begin(); iter != valuesAsString.end(); iter++)
    {
    this->Data->Contours.push_back(std::stod(*iter));
    }
}

// --------------------------------------------------------------------------
int CinemaHelper::GetNumberOfContours()
{
  return this->Data->Contours.size();
}

// --------------------------------------------------------------------------
double CinemaHelper::GetContourValue(int idx)
{
  return this->Data->Contours[idx];
}

// --------------------------------------------------------------------------
int CinemaHelper::GetNumberOfCameraPositions()
{
  return this->Data->NumberOfCameraPositions;
}

// --------------------------------------------------------------------------
#ifdef ENABLE_CATALYST
void CinemaHelper::ApplyCameraPosition(vtkSMViewProxy* view, int cameraPositionIndex)
{
  if (cameraPositionIndex < this->Data->NumberOfCameraPositions && this->Data->CameraPositions)
    {
    this->Data->CurrentCameraPosition = cameraPositionIndex;
    double* cameraPosition = &this->Data->CameraPositions[cameraPositionIndex * 9];

    // std::cout << "Apply camera idx: " << cameraPositionIndex
    //   << "\n fp(" << cameraPosition[0] << ", " << cameraPosition[1] << ", " << cameraPosition[2] << ")"
    //   << "\n position(" << cameraPosition[3] << ", " << cameraPosition[4] << ", " << cameraPosition[5] << ")"
    //   << "\n viewUp(" << cameraPosition[6] << ", " << cameraPosition[7] << ", " << cameraPosition[8] << ")"
    //   << std::endl;

    vtkSMPropertyHelper(view, "CameraFocalPoint").Set(&cameraPosition[0], 3);
    vtkSMPropertyHelper(view, "CameraPosition").Set(&cameraPosition[3], 3);
    vtkSMPropertyHelper(view, "CameraViewUp").Set(&cameraPosition[6], 3);
    view->UpdateVTKObjects();
    }
}
#endif // ENABLE_CATALYST

#if defined(ENABLE_CATALYST) || defined (ENABLE_VTK_RENDERING)
// --------------------------------------------------------------------------
void CinemaHelper::ApplyCameraPosition(vtkCamera *camera, int cameraPositionIndex)
{
  if (cameraPositionIndex < this->Data->NumberOfCameraPositions && this->Data->CameraPositions)
    {
    this->Data->CurrentCameraPosition = cameraPositionIndex;
    double* cameraPosition = &this->Data->CameraPositions[cameraPositionIndex * 9];

    // std::cout << "Apply camera idx: " << cameraPositionIndex
    //   << "\n fp(" << cameraPosition[0] << ", " << cameraPosition[1] << ", " << cameraPosition[2] << ")"
    //   << "\n position(" << cameraPosition[3] << ", " << cameraPosition[4] << ", " << cameraPosition[5] << ")"
    //   << "\n viewUp(" << cameraPosition[6] << ", " << cameraPosition[7] << ", " << cameraPosition[8] << ")"
    //   << std::endl;

    camera->SetFocalPoint(&cameraPosition[0]);
    camera->SetPosition(&cameraPosition[3]);
    camera->SetViewUp(&cameraPosition[6]);
    }
}
#endif

// --------------------------------------------------------------------------
void CinemaHelper::SetScalarAssociation(const std::string& assoc)
{
  this->Data->ScalarAssociation = assoc;
}

// --------------------------------------------------------------------------
std::string CinemaHelper::GetScalarAssociation() const
{
  return this->Data->ScalarAssociation;
}

// --------------------------------------------------------------------------
void CinemaHelper::SetScalarName(const std::string& name)
{
  this->Data->ScalarName = name;
}

// --------------------------------------------------------------------------
std::string CinemaHelper::GetScalarName() const
{
  return this->Data->ScalarName;
}

// --------------------------------------------------------------------------
#ifdef ENABLE_CATALYST
int CinemaHelper::RegisterLayer(const std::string& name, vtkSMRepresentationProxy* representation, double scalarValue)
{
  this->Data->Representations.push_back(representation);

  std::ostringstream layerName;
  layerName << name;
  if (name != "Outline")
    {
    layerName << " ";
    layerName << scalarValue;
    }

  std::ostringstream jsonContent;
  jsonContent
    << "{"                          << endl
    << "  \"colorBy\": ["           << endl
    << "    {"                      << endl
    << "      \"type\": \"const\"," << endl
    << "      \"name\": \"scalar\"," << endl
    << "      \"value\": " << scalarValue << endl
    << "    }"                      << endl
    << "  ],"                       << endl
    << "  \"name\": \"" << layerName.str() << "\"" << endl
    << "}";
  this->Data->JSONPipeline.push_back(jsonContent.str());

  std::ostringstream jsonContent2;
  jsonContent2
    << "{" << endl
    << "  \"name\": \"" << layerName.str() << "\"," << endl
    << "  \"ids\": [\"" << this->Data->LAYER_CODES.substr(this->Data->JSONCompositePipeline.size(), 1) << "\"]" << endl
    << "}";
  this->Data->JSONCompositePipeline.push_back(jsonContent2.str());
  return 1;
}
#endif // ENABLE_CATALYST

#if defined(ENABLE_CATALYST) || defined (ENABLE_VTK_RENDERING)
// --------------------------------------------------------------------------
int CinemaHelper::RegisterLayer(const std::string& name, vtkActor* actor, double scalarValue)
{
  this->Data->Actors.push_back(actor);

  std::ostringstream layerName;
  layerName << name;
  if (name != "Outline")
    {
    layerName << " ";
    layerName << scalarValue;
    }

  std::ostringstream jsonContent;
  jsonContent
    << "{"                          << endl
    << "  \"colorBy\": ["           << endl
    << "    {"                      << endl
    << "      \"type\": \"const\"," << endl
    << "      \"name\": \"scalar\"," << endl
    << "      \"value\": " << scalarValue << endl
    << "    }"                      << endl
    << "  ],"                       << endl
    << "  \"name\": \"" << layerName.str() << "\"" << endl
    << "}";
  this->Data->JSONPipeline.push_back(jsonContent.str());

  std::ostringstream jsonContent2;
  jsonContent2
    << "{" << endl
    << "  \"name\": \"" << layerName.str() << "\"," << endl
    << "  \"ids\": [\"" << this->Data->LAYER_CODES.substr(this->Data->JSONCompositePipeline.size(), 1) << "\"]" << endl
    << "}";
  this->Data->JSONCompositePipeline.push_back(jsonContent2.str());
  return 1;
}
#endif

#ifdef ENABLE_CATALYST
// --------------------------------------------------------------------------
void CinemaHelper::Capture(vtkSMViewProxy* view)
{
  vtkSMRenderViewProxy* renderView = vtkSMRenderViewProxy::SafeDownCast(view);
  if (this->Data->CaptureMethod == "CaptureSortedCompositeData" && renderView)
    {
    this->CaptureSortedCompositeData(renderView);
    }
  else if (this->Data->CaptureMethod == "CaptureImage")
    {
    this->CaptureImage(view, "_.jpg", "vtkJPEGWriter");
    }
}

// --------------------------------------------------------------------------
void CinemaHelper::CaptureSortedCompositeData(vtkSMRenderViewProxy* view)
{
  // Show all representations
  std::vector<vtkSmartPointer<vtkSMRepresentationProxy>>::iterator repIter;
  for (repIter = this->Data->Representations.begin(); repIter != this->Data->Representations.end(); ++repIter)
    {
    vtkSMPropertyHelper(*repIter, "Visibility").Set(1);
    (*repIter)->UpdateVTKObjects();
    }

  // Fix camera bounds
  vtkSMPropertyHelper(view, "LockBounds").Set(0);
  view->UpdateVTKObjects();
  view->StillRender();
  vtkSMPropertyHelper(view, "LockBounds").Set(1);
  view->UpdateVTKObjects();

  // Hide all representations
  for (repIter = this->Data->Representations.begin(); repIter != this->Data->Representations.end(); ++repIter)
    {
    vtkSMPropertyHelper(*repIter, "Visibility").Set(0);
    (*repIter)->UpdateVTKObjects();
    }

  // Show only active Representation
  // Extract images for each fields
  std::vector<vtkSmartPointer<vtkFloatArray>> zBuffers;
  std::vector<vtkSmartPointer<vtkUnsignedCharArray>> luminances;
  size_t compositeSize = this->Data->Representations.size();
  int linearSize = 0;
  for (size_t compositeIdx = 0; compositeIdx < compositeSize; compositeIdx++)
    {
    vtkSMRepresentationProxy* rep = this->Data->Representations[compositeIdx];

    // Hide previous representation
    if (compositeIdx > 0)
      {
      vtkSMRepresentationProxy* previousRep = this->Data->Representations[compositeIdx - 1];
      vtkSMPropertyHelper(previousRep, "Visibility").Set(0);
      previousRep->UpdateVTKObjects();
      }

    // Show current representation
    vtkSMPropertyHelper(rep, "Visibility").Set(1);
    rep->UpdateVTKObjects();

    // capture Z
    view->StillRender();
    vtkSmartPointer<vtkFloatArray> zBuffer = vtkSmartPointer<vtkFloatArray>::New();
    zBuffer->DeepCopy(view->CaptureDepthBuffer());
    // vtkSmartPointer<vtkFloatArray> zBuffer = view->CaptureDepthBuffer(); // FIXME test
    zBuffers.push_back(zBuffer);

    // Prevent color interference to handle light (intensity)
    double white[3] = {1, 1, 1};
    vtkSMPropertyHelper(rep, "DiffuseColor").Set(white, 3);
    vtkSMPropertyHelper(rep, "AmbientColor").Set(white, 3);
    vtkSMPropertyHelper(rep, "SpecularColor").Set(white, 3);
    rep->UpdateVTKObjects();
    view->InvokeCommand("StartCaptureLuminance");
    vtkUnsignedCharArray* imagescalars = vtkUnsignedCharArray::SafeDownCast(view->CaptureWindow(1)->GetPointData()->GetScalars());

    // Extract specular information
    int specularOffset = 1; // [diffuse, specular, ?]
    linearSize = imagescalars->GetNumberOfTuples();
    vtkSmartPointer<vtkUnsignedCharArray> specularComponent = vtkSmartPointer<vtkUnsignedCharArray>::New();
    specularComponent->SetNumberOfComponents(1);
    specularComponent->SetNumberOfTuples(linearSize);

    for (int idx = 0; idx < linearSize; idx++)
      {
      specularComponent->SetValue(idx, imagescalars->GetValue(idx * 3 + specularOffset));
      }

    luminances.push_back(specularComponent);
    view->InvokeCommand("StopCaptureLuminance");
    }

  // Post process arrays to write proper data structure
  if (!this->Data->IsRoot)
    {
    // Skip post processing if not root...
    return;
    }

  int stackSize = linearSize * compositeSize;
  vtkNew<vtkUnsignedCharArray> orderArray;
  orderArray->SetNumberOfComponents(1);
  orderArray->SetNumberOfTuples(stackSize);
  std::vector<Pixel> pixelSorter;
  while (pixelSorter.size() < compositeSize)
    {
    pixelSorter.push_back(Pixel());
    }

  for (int pixelId = 0; pixelId < linearSize; pixelId++)
    {
    // Fill pixelSorter
    for (size_t layerIdx = 0; layerIdx < compositeSize; layerIdx++)
      {
      float depth = zBuffers[layerIdx]->GetValue(pixelId);
      if (depth < 1.0)
        {
        pixelSorter[layerIdx].Index = (unsigned char)layerIdx;
        pixelSorter[layerIdx].Depth = depth;
        }
      else
        {
        pixelSorter[layerIdx].Index = 255;
        pixelSorter[layerIdx].Depth = 1.0;
        }
      }

    // Sort pixels
    std::sort(pixelSorter.begin(), pixelSorter.end(), pixelComp);

    // Fill sortedOrder array
    for (size_t layerIdx = 0; layerIdx < compositeSize; layerIdx++)
      {
      orderArray->SetValue(layerIdx * linearSize + pixelId, pixelSorter[layerIdx].Index);
      }
    }

  // Write order file
  std::string orderFileName = this->Data->getDataAbsoluteFilePath("order.uint8", true);
  std::ofstream fp(orderFileName.c_str(), ios::out | ios::binary);

  if (fp.fail())
    {
    std::cout << "Unable to open file: "<< orderFileName.c_str() << std::endl;
    }
  else
    {
    fp.write((char*)orderArray->GetPointer(0), stackSize);
    fp.flush();
    fp.close();
    }

  // Compute intensity data
  vtkNew<vtkUnsignedCharArray> intensityArray;
  intensityArray->SetNumberOfComponents(1);
  intensityArray->SetNumberOfTuples(stackSize);
  for (int idx = 0; idx < stackSize; idx++)
    {
    int layerIdx = orderArray->GetValue(idx);
    if (layerIdx < 255)
      {
      intensityArray->SetValue(idx, luminances[layerIdx]->GetValue(idx % linearSize));
      }
    else
      {
      intensityArray->SetValue(idx, 0);
      }

    }

  // Write light intensity file
  std::string intensityFileName = this->Data->getDataAbsoluteFilePath("intensity.uint8", true);
  std::ofstream fpItensity(intensityFileName.c_str(), ios::out | ios::binary);

  if (fpItensity.fail())
    {
    std::cout << "Unable to open file: "<< intensityFileName.c_str() << std::endl;
    }
  else
    {
    fpItensity.write((char*)intensityArray->GetPointer(0), stackSize);
    fpItensity.flush();
    fpItensity.close();
    }
}
#endif // ENABLE_CATALYST

#if defined(ENABLE_CATALYST) || defined (ENABLE_VTK_RENDERING)
// --------------------------------------------------------------------------
void CinemaHelper::Render(vtkRenderWindow* renderWindow)
{
  renderWindow->SetSize(this->Data->ImageSize[0], this->Data->ImageSize[1]);
#if defined(ENABLE_CATALYST) || defined (ENABLE_VTK_MPI)
  vtkMultiProcessController* controller = vtkMultiProcessController::GetGlobalController();
  if (controller->GetLocalProcessId() == 0)
    {
    renderWindow->Render();
    controller->TriggerBreakRMIs();
    controller->Barrier();
    }
  else
    {
    controller->ProcessRMIs();
    controller->Barrier();
    }
#else
  renderWindow->Render();
#endif
}
#endif

#if defined(ENABLE_CATALYST) || defined (ENABLE_VTK_RENDERING)
vtkImageData* CinemaHelper::CaptureWindow(vtkRenderWindow* renderWindow)
{
  int swapBuffers = renderWindow->GetSwapBuffers();
  renderWindow->SwapBuffersOff();

  this->Render(renderWindow);

  vtkNew<vtkWindowToImageFilter> w2i;
  w2i->SetInput(renderWindow);
  w2i->SetScale(1, 1);
  w2i->ReadFrontBufferOff();
  w2i->ShouldRerenderOff(); // WindowToImageFilter can re-render as needed too,
                            // we just don't require the first render.

  // Note how we simply called `Update` here. Since `WindowToImageFilter` calls
  // this->RenderForImageCapture() we don't have to worry too much even if it
  // gets called only on the client side (or root node in batch mode).
  w2i->Update();

  renderWindow->SetSwapBuffers(swapBuffers);

  vtkImageData* capture = vtkImageData::New();
  capture->ShallowCopy(w2i->GetOutput());
  return capture;
}
#endif

#ifdef ENABLE_CATALYST
// --------------------------------------------------------------------------
void CinemaHelper::CaptureSortedCompositeData(vtkRenderWindow* renderWindow, vtkRenderer* renderer,
  vtkIceTCompositePass* compositePass)
{
  vtkSynchronizedRenderers::vtkRawImage image;
  vtkMultiProcessController* controller = vtkMultiProcessController::GetGlobalController();
  // Show all representations
  std::vector<vtkSmartPointer<vtkActor>>::iterator actorIter;
  for (actorIter = this->Data->Actors.begin(); actorIter != this->Data->Actors.end(); ++actorIter)
    {
    (*actorIter)->SetVisibility(1);
    }

  // Fix camera bounds
  this->Render(renderWindow);
  double prop_bounds[6];
  if (this->Data->IsRoot)
  {
    renderer->ComputeVisiblePropBounds(prop_bounds);
  }
  controller->Broadcast(prop_bounds, 6, 0);

  // Hide all representations
  for (actorIter = this->Data->Actors.begin(); actorIter != this->Data->Actors.end(); ++actorIter)
    {
    (*actorIter)->SetVisibility(0);
    }

  // Show only active Representation
  // Extract images for each fields
  std::vector<vtkSmartPointer<vtkFloatArray>> zBuffers;
  std::vector<vtkSmartPointer<vtkUnsignedCharArray>> luminances;
  size_t compositeSize = this->Data->Actors.size();
  size_t linearSize = 0;
  for (size_t compositeIdx = 0; compositeIdx < compositeSize; compositeIdx++)
    {
    vtkActor* actor = this->Data->Actors[compositeIdx];

    // Hide previous representation
    if (compositeIdx > 0)
      {
      vtkActor* previousActor = this->Data->Actors[compositeIdx - 1];
      previousActor->SetVisibility(0);
      }

    // Show current representation
    actor->SetVisibility(1);

    // capture Z
    renderer->ResetCameraClippingRange(prop_bounds);
    this->Render(renderWindow);

    if (this->Data->IsRoot)
      {
      vtkSmartPointer<vtkFloatArray> zBuffer = vtkSmartPointer<vtkFloatArray>::New();

      zBuffer->DeepCopy(compositePass->GetLastRenderedDepths());
      // cout << "PID(" << controller->GetLocalProcessId()
      //     << "): Size " << zBuffer->GetNumberOfTuples()
      //     << " Range " << zBuffer->GetRange()[0] << ", " << zBuffer->GetRange()[1] << endl;
      zBuffers.push_back(zBuffer);

      linearSize = zBuffer->GetNumberOfTuples();

      // ------------------------------------------------------------------------
      // Capture Luminance
      // ------------------------------------------------------------------------
      compositePass->GetLastRenderedTile(image);
      vtkUnsignedCharArray* imagescalars = image.GetRawPtr();
      size_t step = imagescalars->GetNumberOfComponents();
      vtkSmartPointer<vtkUnsignedCharArray> specularComponent = vtkSmartPointer<vtkUnsignedCharArray>::New();
      specularComponent->SetNumberOfComponents(1);
      specularComponent->SetNumberOfTuples(linearSize);
      for (size_t idx = 0; idx < linearSize; idx++)
        {
        specularComponent->SetValue(idx, imagescalars->GetValue(idx * step));
        }

      luminances.push_back(specularComponent);
      }
    }

  // Post process arrays to write proper data structure
  if (!this->Data->IsRoot)
    {
    // Skip post processing if not root...
    return;
    }

  size_t stackSize = linearSize * compositeSize;
  vtkNew<vtkUnsignedCharArray> orderArray;
  orderArray->SetNumberOfComponents(1);
  orderArray->SetNumberOfTuples(stackSize);
  std::vector<Pixel> pixelSorter;
  while (pixelSorter.size() < compositeSize)
    {
    pixelSorter.push_back(Pixel());
    }

  for (size_t pixelId = 0; pixelId < linearSize; pixelId++)
    {
    // Fill pixelSorter
    for (size_t layerIdx = 0; layerIdx < compositeSize; layerIdx++)
      {
      float depth = zBuffers[layerIdx]->GetValue(pixelId);
      if (depth < 1.0)
        {
        pixelSorter[layerIdx].Index = (unsigned char)layerIdx;
        pixelSorter[layerIdx].Depth = depth;
        }
      else
        {
        pixelSorter[layerIdx].Index = 255;
        pixelSorter[layerIdx].Depth = 1.0;
        }
      }

    // Sort pixels
    std::sort(pixelSorter.begin(), pixelSorter.end(), pixelComp);

    // Fill sortedOrder array
    for (size_t layerIdx = 0; layerIdx < compositeSize; layerIdx++)
      {
      orderArray->SetValue(layerIdx * linearSize + pixelId, pixelSorter[layerIdx].Index);
      }
    }

  // Write order file
  std::string orderFileName = this->Data->getDataAbsoluteFilePath("order.uint8", true);
  std::ofstream fp(orderFileName.c_str(), ios::out | ios::binary);

  if (fp.fail())
    {
    std::cout << "Unable to open file: "<< orderFileName.c_str() << std::endl;
    }
  else
    {
    fp.write((char*)orderArray->GetPointer(0), stackSize);
    fp.flush();
    fp.close();
    }

  // Compute intensity data
  vtkNew<vtkUnsignedCharArray> intensityArray;
  intensityArray->SetNumberOfComponents(1);
  intensityArray->SetNumberOfTuples(stackSize);
  for (size_t idx = 0; idx < stackSize; idx++)
    {
    size_t layerIdx = orderArray->GetValue(idx);
    if (layerIdx < 255)
      {
      intensityArray->SetValue(idx, luminances[layerIdx]->GetValue(idx % linearSize));
      }
    else
      {
      intensityArray->SetValue(idx, 0);
      }

    }

  // Write light intensity file
  std::string intensityFileName = this->Data->getDataAbsoluteFilePath("intensity.uint8", true);
  std::ofstream fpItensity(intensityFileName.c_str(), ios::out | ios::binary);

  if (fpItensity.fail())
    {
    std::cout << "Unable to open file: "<< intensityFileName.c_str() << std::endl;
    }
  else
    {
    fpItensity.write((char*)intensityArray->GetPointer(0), stackSize);
    fpItensity.flush();
    fpItensity.close();
    }
}

// --------------------------------------------------------------------------
void CinemaHelper::CaptureImage(vtkSMViewProxy* view, const std::string fileName, const std::string writerName, double scale, bool createDirectory)
{
  view->WriteImage(this->Data->getDataAbsoluteFilePath(fileName, createDirectory).c_str(), writerName.c_str(), scale);
}
#endif // ENABLE_CATALYST

// --------------------------------------------------------------------------
void CinemaHelper::WriteVolume(vtkImageData* image)
{
  std::string jsonName = "volume.json";
  std::string dataName = "volume.data";

  if (!this->Data->IsRoot)
    {
      std::ostringstream json;
      json << "volume_" << this->Data->PID << ".json";
      jsonName = json.str();

      std::ostringstream data;
      data << "volume_" << this->Data->PID << ".data";
      dataName = data.str();
    }

  if (image == nullptr)
    {
    return;
    }

  vtkFloatArray* array = vtkFloatArray::SafeDownCast(image->GetPointData()->GetScalars());

  // Write volume.json
  std::string metaFileName = this->Data->getDataAbsoluteFilePath(jsonName, true);
  std::ofstream jsonFilePointer(metaFileName.c_str(), ios::out);
  if (jsonFilePointer.fail())
    {
    std::cout << "Unable to open file: "<< metaFileName.c_str() << std::endl;
    }
  else
    {
    int extent[6];
    image->GetExtent(extent);
    jsonFilePointer << "{" << endl;
    jsonFilePointer << "    \"origin\": [0, 0, 0]," << endl;
    jsonFilePointer << "    \"spacing\": [1, 1, 1]," << endl;
    jsonFilePointer << "    \"extent\": ["
       << extent[0] << ", " << (extent[1]) << ", "
       << extent[2] << ", " << (extent[3]) << ", "
       << extent[4] << ", " << (extent[5]) << "]," << endl;
    jsonFilePointer << "    \"vtkClass\": \"vtkImageData\"," << endl;
    jsonFilePointer << "    \"pointData\": {" << endl;
    jsonFilePointer << "        \"vtkClass\": \"vtkDataSetAttributes\"," << endl;
    jsonFilePointer << "        \"arrays\": [{" << endl;
    jsonFilePointer << "            \"data\": {" << endl;
    jsonFilePointer << "                \"numberOfComponents\": 1," << endl;
    jsonFilePointer << "                \"name\": \"scalars\"," << endl;
    jsonFilePointer << "                \"vtkClass\": \"vtkDataArray\"," << endl;
    jsonFilePointer << "                \"dataType\": \"Float32Array\"," << endl;
    jsonFilePointer << "                \"ref\": {" << endl;
    jsonFilePointer << "                    \"registration\": \"setScalars\"," << endl;
    jsonFilePointer << "                    \"encode\": \"LittleEndian\"," << endl;
    jsonFilePointer << "                    \"basepath\": \"" << this->Data->NumberOfTimeSteps << "\"," << endl;
    jsonFilePointer << "                    \"id\": \"volume.data\"" << endl;
    jsonFilePointer << "                }," << endl;
    jsonFilePointer << "                \"size\": "<< array->GetNumberOfValues() << endl;
    jsonFilePointer << "            }" << endl;
    jsonFilePointer << "        }]" << endl;
    jsonFilePointer << "    }" << endl;
    jsonFilePointer << "}" << endl;
    jsonFilePointer.flush();
    jsonFilePointer.close();
    }

  // Write volume.data
  std::string dataFileName = this->Data->getDataAbsoluteFilePath(dataName, true);
  // std::cout << "write volume: " << dataFileName.c_str() << std::endl;
  std::ofstream filePointer(dataFileName.c_str(), ios::out | ios::binary);

  if (filePointer.fail())
    {
    std::cout << "Unable to open file: "<< dataFileName.c_str() << std::endl;
    }
  else
    {
    int stackSize = array->GetNumberOfTuples() * 4;
    filePointer.write((char*)array->GetVoidPointer(0), stackSize);
    filePointer.flush();
    filePointer.close();
    }
}

void CinemaHelper::WriteCDF(long long totalArraySize, const double* cdfValues)
{
  if (!this->Data->IsRoot)
  {
    return;
  }
  std::string dataName = "cdf.float32";
  std::string dataFilePath = this->Data->getDataAbsoluteFilePath(dataName, true);
  std::ofstream dataFilePathPointer(dataFilePath.c_str(), std::ios::out | std::ios::binary);
  if (dataFilePathPointer.fail())
  {
    std::cout << "Unable to open file: " << dataFilePath.c_str() << std::endl;
  }
  else
  {
    long long stackSize = this->Data->SampleSize * 4;
    std::vector<float> cdfFloats(cdfValues, cdfValues + this->Data->SampleSize);
    /*
    float* cdfFloats = new float[cdfSize];
    for (int i = 0; i < cdfSize; i++)
    {
    cdfFloats[i] = (float)cdfValues[i];
    }

    dataFilePathPointer.write((char*)cdfFloats, stackSize);
    */
    dataFilePathPointer.write((char*)&cdfFloats[0], stackSize);
    dataFilePathPointer.flush();
    dataFilePathPointer.close();
    this->Data->JSONData["cdf"] =
      "{\n"
      "    \"pattern\": \"{time}/cdf.float32\",\n"
      "    \"name\": \"cdf\",\n"
      "    \"type\": \"arraybuffer\"\n"
      "}\n"
      ;
    std::ostringstream xmeta;
    xmeta << "   ,\"totalCount\": " << totalArraySize;
    this->Data->JSONExtraMetadata = xmeta.str();
    //delete[] cdfFloats;
  }
}

}
