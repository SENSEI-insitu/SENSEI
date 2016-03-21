#include "ImageProperties.h"

namespace sensei
{
namespace libsim
{

ImageProperties::ImageProperties() : filename("image%ts.png"),
    format("png"), width(1920), height(1080)
{
}

ImageProperties::~ImageProperties()
{
}

void ImageProperties::SetFilename(const std::string &s)
{
   filename = s;
}

void ImageProperties::SetWidth(int val)
{
   if(val > 0) width = val;
}

void ImageProperties::SetHeight(int val)
{
    if(val > 0) height = val;
}

void ImageProperties::SetFormat(const std::string &s)
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

const std::string &ImageProperties::GetFilename() const
{ return filename; }

const std::string &ImageProperties::GetFormat() const
{ return format; }

int ImageProperties::GetWidth() const
{ return width; }

int ImageProperties::GetHeight() const
{ return height; }
    
    
} // libsim
} // sensei
