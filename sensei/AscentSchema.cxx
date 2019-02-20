#include "AscentSchema.h"
#include "Error.h"

//
// Great information on the following web page for the pipeline actions.
// https://ascent.readthedocs.io/en/latest/Actions/Pipelines.html
//

namespace sensei
{

// --------------------------------------------------------------------------
static int Get3Slice(pugi::xml_node &node, conduit::Node &actions)
{
  actions["pipelines/pl1/f1/type"] = "3slice";

  std::string field;
  if(node.attribute("plotvars"))
  {
    field = node.attribute("plotvars").value();
    actions["pipelines/pl1/f1/params/field"] = field;
  }
  else
  {
    SENSEI_ERROR("add_pipelines action must have field defined");
    return( -1 );
  }

  double x_offset = 0.0, y_offset = 0.0, z_offset = 0.0;
  if(node.attribute("offset"))
    sscanf(node.attribute("offset").value(), "%lg,%lg,%lg", &x_offset, &y_offset, &z_offset);

  actions["pipelines/pl1/f1/params/x_offset"] = x_offset;
  actions["pipelines/pl1/f1/params/y_offset"] = y_offset;
  actions["pipelines/pl1/f1/params/z_offset"] = z_offset;

  return( 0 );
}

// --------------------------------------------------------------------------
static int GetSlice(pugi::xml_node &node, conduit::Node &actions)
{
  actions["pipelines/pl1/f1/type"] = "slice";

  std::string field;
  if(node.attribute("plotvars"))
  {
    field = node.attribute("plotvars").value();
    actions["pipelines/pl1/f1/params/field"] = field;
  }
  else
  {
    SENSEI_ERROR("add_pipelines action must have field defined");
    return( -1 );
  }

  double x = 0.0, y = 0.0, z = 0.0;
  double nx = 0.0, ny = 0.0, nz = 1.0;

  if(node.attribute("plane"))
    sscanf(node.attribute("plane").value(), "%lg,%lg,%lg", &x, &y, &z);
  if(node.attribute("normal"))
    sscanf(node.attribute("normal").value(), "%lg,%lg,%lg", &nx, &ny, &nz);

  actions["pipelines/pl1/f1/params/point/x"] = x;
  actions["pipelines/pl1/f1/params/point/y"] = y;
  actions["pipelines/pl1/f1/params/point/z"] = z;

  actions["pipelines/pl1/f1/params/normal/x"] = nx;
  actions["pipelines/pl1/f1/params/normal/y"] = ny;
  actions["pipelines/pl1/f1/params/normal/z"] = nz;

  return( 0 );
}

// --------------------------------------------------------------------------
static int GetClip(pugi::xml_node &node, conduit::Node &actions)
{
  actions["pipelines/pl1/f1/type"] = "clip";

  std::string field;
  if(node.attribute("plotvars"))
  {
    field = node.attribute("plotvars").value();
    actions["pipelines/pl1/f1/params/field"] = field;
  }
  else
  {
    SENSEI_ERROR("add_pipelines action must have field defined");
    return( -1 );
  }

  actions["pipelines/pl1/f1/params/topology"] = "mesh";

  std::string shape  = "sphere"; 
  std::string invert = "false";
  if(node.attribute("shape"))
    shape = node.attribute("shape").value();

  if(node.attribute("invert"))
    invert = node.attribute("invert").value();
  actions["pipelines/pl1/f1/params/invert"] = invert;

  if(shape == "sphere")
  {
    double radius = 5.0;
    if(node.attribute("radius"))
      sscanf(node.attribute("radius").value(), "%lg", &radius);
    actions["pipelines/pl1/f1/params/sphere/radius"] = radius;

    double x = 0.0, y = 0.0, z = 0.0;
    if(node.attribute("center"))
      sscanf(node.attribute("center").value(), "%lg,%lg,%lg", &x, &y, &z);
    actions["pipelines/pl1/f1/params/sphere/center/x"] = x;
    actions["pipelines/pl1/f1/params/sphere/center/y"] = y;
    actions["pipelines/pl1/f1/params/sphere/center/z"] = z;
  }
  else if(shape == "box")
  {
    double x_max = 10.0, y_max = 10.0, z_max = 10.0;
    double x_min = 0.0, y_min = 0.0, z_min = 0.0;
    if(node.attribute("box_min"))
      sscanf(node.attribute("box_min").value(), "%lg,%lg,%lg", &x_min, &y_min, &z_min);
    if(node.attribute("box_max"))
      sscanf(node.attribute("box_max").value(), "%lg,%lg,%lg", &x_max, &y_max, &z_max);

    actions["pipelines/pl1/f1/params/box/min/x"] = x_min;
    actions["pipelines/pl1/f1/params/box/min/y"] = y_min;
    actions["pipelines/pl1/f1/params/box/min/z"] = z_min;

    actions["pipelines/pl1/f1/params/box/max/x"] = x_max;
    actions["pipelines/pl1/f1/params/box/max/y"] = y_max;
    actions["pipelines/pl1/f1/params/box/max/z"] = z_max;
  }
  else if(shape == "plane")
  {
    double x = 0.0, y = 0.0, z = 0.0;
    double nx = 0.0, ny = 0.0, nz = 1.0;

    if(node.attribute("plane"))
      sscanf(node.attribute("plane").value(), "%lg,%lg,%lg", &x, &y, &z);
    if(node.attribute("normal"))
      sscanf(node.attribute("normal").value(), "%lg,%lg,%lg", &nx, &ny, &nz);

    actions["pipelines/pl1/f1/params/plane/point/x"] = x;
    actions["pipelines/pl1/f1/params/plane/point/y"] = y;
    actions["pipelines/pl1/f1/params/plane/point/z"] = z;

    actions["pipelines/pl1/f1/params/plane/normal/x"] = nx;
    actions["pipelines/pl1/f1/params/plane/normal/y"] = ny;
    actions["pipelines/pl1/f1/params/plane/normal/z"] = nz;
  }
  else
  {
    SENSEI_ERROR("Clip must specify a sphere, box, or plane");
    return( -1 );
  }

  return( 0 );
}

// --------------------------------------------------------------------------
static int GetClipWithField(pugi::xml_node &node, conduit::Node &actions)
{
  actions["pipelines/pl1/f1/type"] = "clip_with_field";

  std::string field;
  if(node.attribute("plotvars"))
  {
    field = node.attribute("plotvars").value();
    actions["pipelines/pl1/f1/params/field"] = field;
  }
  else
  {
    SENSEI_ERROR("add_pipelines action must have field defined");
    return( -1 );
  }

  std::string invert = "false";
  double clip_value = 0.0;
  if(node.attribute("invert"))
    invert = node.attribute("invert").value();
  if(node.attribute("clip_value"))
    sscanf(node.attribute("clip_value").value(), "%lg", &clip_value);

  actions["pipelines/pl1/f1/params/clip_value"] = clip_value;
  actions["pipelines/pl1/f1/params/invert"]     = invert;

  return( 0 );
}

// --------------------------------------------------------------------------
static int GetThreshold(pugi::xml_node &node, conduit::Node &actions)
{
  actions["pipelines/pl1/f1/type"] = "threshold";

  std::string field;
  if(node.attribute("plotvars"))
  {
    field = node.attribute("plotvars").value();
    actions["pipelines/pl1/f1/params/field"] = field;
  }
  else
  {
    SENSEI_ERROR("add_pipelines action must have field defined");
    return( -1 );
  }
  if(node.attribute("min_value") && node.attribute("max_value"))
  {
    double min = 0.0, max = 1.0;
    if(node.attribute("min_value"))
      sscanf(node.attribute("min_value").value(), "%lg", &min);
    if(node.attribute("max_value"))
      sscanf(node.attribute("max_value").value(), "%lg", &max);
    actions["pipelines/pl1/f1/params/min_value"] = min;
    actions["pipelines/pl1/f1/params/max_value"] = max;
  }
  else
  {
    SENSEI_ERROR("Threshold requires a min_value and max_value defined");
    return( -1 );
  }
  return( 0 );
}

// --------------------------------------------------------------------------
static int GetIsoVolume(pugi::xml_node &node, conduit::Node &actions)
{
  actions["pipelines/pl1/f1/type"] = "iso_volume";

  std::string field;
  if(node.attribute("plotvars"))
  {
    field = node.attribute("plotvars").value();
    actions["pipelines/pl1/f1/params/field"] = field;
  }
  else
  {
    SENSEI_ERROR("add_pipelines action must have field defined");
    return( -1 );
  }

  double min_value = 0.0, max_value = 10.0;
  if(node.attribute("min_value"))
    sscanf(node.attribute("min_value").value(), "%lg", &min_value);
  if(node.attribute("max_value"))
    sscanf(node.attribute("max_value").value(), "%lg", &max_value);

  actions["pipelines/pl1/f1/params/min_value"] = min_value;
  actions["pipelines/pl1/f1/params/max_value"] = max_value;

  return( 0 );
}

// --------------------------------------------------------------------------
static int GetContour(pugi::xml_node &node, conduit::Node &actions)
{
  actions["pipelines/pl1/f1/type"] = "contour";

  std::string field;
  if(node.attribute("plotvars"))
  {
    field = node.attribute("plotvars").value();
    actions["pipelines/pl1/f1/params/field"] = field;
  }
  else
  {
    SENSEI_ERROR("add_pipelines action must have field defined");
    return( -1 );
  }
  if(node.attribute("value") || node.attribute("levels"))
  {
    if(node.attribute("value"))
    {
      double isoval;
      sscanf(node.attribute("value").value(), "%lg", &isoval);
      actions["pipelines/pl1/f1/params/iso_values"] = isoval;
    }
    else
    {
      int levels;
      sscanf(node.attribute("levels").value(), "%d", &levels);
      actions["pipelines/pl1/f1/params/levels"] = levels;
    }
  }
  else
  {
    SENSEI_ERROR("contour pipeline must have values or levels defined");
    return( -1 );
  }
  return( 0 );
}

// --------------------------------------------------------------------------
int AscentGetScene(pugi::xml_node &node, conduit::Node &actions)
{
  actions["action"] = "add_scenes";

  std::string plot;
  if(node.attribute("plot"))
  {
    plot = node.attribute("plot").value();
    actions["scenes/scene1/plots/plt1/type"] = plot;
  }
  else
  {
    SENSEI_ERROR("add_scenes action must have plot defined");
    return( -1 );
  }

  std::string field;
  if(node.attribute("plotvars"))
  {
    field = node.attribute("plotvars").value();
    actions["scenes/scene1/plots/plt1/field"] = field;
  }
  else
  {
    SENSEI_ERROR("add_scenes action must have field defined");
    return( -1 );
  }
  return( 0 );
}

// --------------------------------------------------------------------------
int AscentGetPipeline(pugi::xml_node &node, conduit::Node &actions)
{
  actions["action"] = "add_pipelines";

  std::string pipeline;
  if(node.attribute("pipeline"))
  {
    pipeline = node.attribute("pipeline").value();
  }
  else
  {
    SENSEI_ERROR("add_pipelines action must have pipeline defined");
    return( -1 );
  }

  if(pipeline == "contour")
  {
    if(GetContour(node, actions))
      return( -1 );
  }
  else if(pipeline == "threshold")
  {
    if(GetThreshold(node, actions))
      return( -1 );
  }
  else if(pipeline == "slice")
  {
    if(GetSlice(node, actions))
      return( -1 );
  }
  else if(pipeline == "3slice")
  {
    if(Get3Slice(node, actions))
      return( -1 );
  }
  else if(pipeline == "clip")
  {
    if(GetClip(node, actions))
      return( -1 );
  }
  else if(pipeline == "clip_with_field")
  {
    if(GetClipWithField(node, actions))
      return( -1 );
  }
  else if(pipeline == "iso_volume")
  {
    if(GetIsoVolume(node, actions))
      return( -1 );
  }
  else
  {
    SENSEI_ERROR("Pipeline "  << pipeline << " is not a supported pipeline");
    return( -1 );
  }
  return( 0 );
}

}
