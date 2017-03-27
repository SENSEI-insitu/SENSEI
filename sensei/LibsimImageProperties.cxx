#include "LibsimImageProperties.h"

namespace sensei
{

LibsimImageProperties::LibsimImageProperties() :
    filename("image%ts.png"), format("png"), width(1920), height(1080)
{
}

LibsimImageProperties::~LibsimImageProperties()
{
}

void LibsimImageProperties::SetFilename(const std::string &s)
{
   filename = s;
}

void LibsimImageProperties::SetWidth(int val)
{
   if(val > 0) width = val;
}

void LibsimImageProperties::SetHeight(int val)
{
    if(val > 0) height = val;
}

void LibsimImageProperties::SetFormat(const std::string &s)
{
    if(s == "bmp" || s == "BMP")
       format = "bmp";
    else if(s == "jpeg" || s == "jpg" || s == "JPEG" || s == "JPG")
       format = "jpeg";
    else if(s == "png" || s == "PNG")
       format = "png";
    else if(s == "ppm" || s == "PPM")
       format = "ppm";
    else if(s == "rgb" || s == "RGB")
       format = "rgb";
    else if(s == "tif" || s == "TIF" || s == "tiff" || s == "TIFF")
       format = "tiff";
}

const std::string &LibsimImageProperties::GetFilename() const
{ return filename; }

const std::string &LibsimImageProperties::GetFormat() const
{ return format; }

int LibsimImageProperties::GetWidth() const
{ return width; }

int LibsimImageProperties::GetHeight() const
{ return height; }

}
