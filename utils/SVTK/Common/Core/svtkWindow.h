/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkWindow.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkWindow
 * @brief   window superclass for svtkRenderWindow
 *
 * svtkWindow is an abstract object to specify the behavior of a
 * rendering window.  It contains svtkViewports.
 *
 * @sa
 * svtkRenderWindow svtkViewport
 */

#ifndef svtkWindow_h
#define svtkWindow_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

class svtkUnsignedCharArray;

class SVTKCOMMONCORE_EXPORT svtkWindow : public svtkObject
{
public:
  svtkTypeMacro(svtkWindow, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * These are window system independent methods that are used
   * to help interface svtkWindow to native windowing systems.
   */
  virtual void SetDisplayId(void*) {}
  virtual void SetWindowId(void*) {}
  virtual void SetParentId(void*) {}
  virtual void* GetGenericDisplayId() { return nullptr; }
  virtual void* GetGenericWindowId() { return nullptr; }
  virtual void* GetGenericParentId() { return nullptr; }
  virtual void* GetGenericContext() { return nullptr; }
  virtual void* GetGenericDrawable() { return nullptr; }
  virtual void SetWindowInfo(const char*) {}
  virtual void SetParentInfo(const char*) {}
  //@}

  //@{
  /**
   * Get the position (x and y) of the rendering window in
   * screen coordinates (in pixels).
   */
  virtual int* GetPosition() SVTK_SIZEHINT(2);

  /**
   * Set the position (x and y) of the rendering window in
   * screen coordinates (in pixels). This resizes the operating
   * system's view/window and redraws it.
   */
  virtual void SetPosition(int x, int y);
  virtual void SetPosition(int a[2]);
  //@}

  //@{
  /**
   * Get the size (width and height) of the rendering window in
   * screen coordinates (in pixels).
   */
  virtual int* GetSize() SVTK_SIZEHINT(2);

  /**
   * Set the size (width and height) of the rendering window in
   * screen coordinates (in pixels). This resizes the operating
   * system's view/window and redraws it.
   *
   * If the size has changed, this method will fire
   * svtkCommand::WindowResizeEvent.
   */
  virtual void SetSize(int width, int height);
  virtual void SetSize(int a[2]);
  //@}

  /**
   * GetSize() returns the size * this->TileScale, whereas this method returns
   * the size without multiplying with the tile scale. Measured in pixels.
   */
  int* GetActualSize() SVTK_SIZEHINT(2);

  /**
   * Get the current size of the screen in pixels.
   */
  virtual int* GetScreenSize() SVTK_SIZEHINT(2) { return nullptr; }

  //@{
  /**
   * Keep track of whether the rendering window has been mapped to screen.
   */
  svtkGetMacro(Mapped, svtkTypeBool);
  //@}

  //@{
  /**
   * Show or not Show the window
   */
  svtkGetMacro(ShowWindow, bool);
  svtkSetMacro(ShowWindow, bool);
  svtkBooleanMacro(ShowWindow, bool);
  //@}

  //@{
  /**
   * Render to an offscreen destination such as a framebuffer.
   * All four combinations of ShowWindow and UseOffScreenBuffers
   * should work for most rendering backends.
   */
  svtkSetMacro(UseOffScreenBuffers, bool);
  svtkGetMacro(UseOffScreenBuffers, bool);
  svtkBooleanMacro(UseOffScreenBuffers, bool);
  //@}

  //@{
  /**
   * Turn on/off erasing the screen between images. This allows multiple
   * exposure sequences if turned on. You will need to turn double
   * buffering off or make use of the SwapBuffers methods to prevent
   * you from swapping buffers between exposures.
   */
  svtkSetMacro(Erase, svtkTypeBool);
  svtkGetMacro(Erase, svtkTypeBool);
  svtkBooleanMacro(Erase, svtkTypeBool);
  //@}

  //@{
  /**
   * Keep track of whether double buffering is on or off
   */
  svtkSetMacro(DoubleBuffer, svtkTypeBool);
  svtkGetMacro(DoubleBuffer, svtkTypeBool);
  svtkBooleanMacro(DoubleBuffer, svtkTypeBool);
  //@}

  //@{
  /**
   * Get name of rendering window
   */
  svtkGetStringMacro(WindowName);
  svtkSetStringMacro(WindowName);
  //@}

  /**
   * Ask each viewport owned by this Window to render its image and
   * synchronize this process.
   */
  virtual void Render() {}

  /**
   * Release any graphics resources that are being consumed by this texture.
   * The parameter window could be used to determine which graphic
   * resources to release.
   */
  virtual void ReleaseGraphicsResources(svtkWindow*) {}

  //@{
  /**
   * Get the pixel data of an image, transmitted as RGBRGBRGB. The
   * front argument indicates if the front buffer should be used or the back
   * buffer. It is the caller's responsibility to delete the resulting
   * array. It is very important to realize that the memory in this array
   * is organized from the bottom of the window to the top. The origin
   * of the screen is in the lower left corner. The y axis increases as
   * you go up the screen. So the storage of pixels is from left to right
   * and from bottom to top.
   * (x,y) is any corner of the rectangle. (x2,y2) is its opposite corner on
   * the diagonal.
   */
  virtual unsigned char* GetPixelData(
    int /*x*/, int /*y*/, int /*x2*/, int /*y2*/, int /*front*/, int /*right*/ = 0)
  {
    return nullptr;
  }
  virtual int GetPixelData(int /*x*/, int /*y*/, int /*x2*/, int /*y2*/, int /*front*/,
    svtkUnsignedCharArray* /*data*/, int /*right*/ = 0)
  {
    return 0;
  }
  //@}

  //@{
  /**
   * Return a best estimate to the dots per inch of the display
   * device being rendered (or printed).
   */
  svtkGetMacro(DPI, int);
  svtkSetClampMacro(DPI, int, 1, SVTK_INT_MAX);
  //@}

  /**
   * Attempt to detect and set the DPI of the display device by querying the
   * system. Note that this is not supported on most backends, and this method
   * will return false if the DPI could not be detected. Use GetDPI() to
   * inspect the detected value.
   */
  virtual bool DetectDPI() { return false; }

  //@{
  /**
   * Convenience to set SHowWindow and UseOffScreenBuffers in one call
   */
  void SetOffScreenRendering(svtkTypeBool val)
  {
    this->SetShowWindow(val == 0);
    this->SetUseOffScreenBuffers(val != 0);
  }
  svtkBooleanMacro(OffScreenRendering, svtkTypeBool);
  //@}

  /**
   * Deprecated, directly use GetShowWindow and GetOffScreenBuffers
   * instead.
   */
  svtkTypeBool GetOffScreenRendering() { return this->GetShowWindow() ? 0 : 1; }

  /**
   * Make the window current. May be overridden in subclasses to do
   * for example a glXMakeCurrent or a wglMakeCurrent.
   */
  virtual void MakeCurrent() {}

  //@{
  /**
   * These methods are used by svtkWindowToImageFilter to tell a SVTK window
   * to simulate a larger window by tiling. For 3D geometry these methods
   * have no impact. It is just in handling annotation that this information
   * must be available to the mappers and the coordinate calculations.
   */
  svtkSetVector2Macro(TileScale, int);
  svtkGetVector2Macro(TileScale, int);
  void SetTileScale(int s) { this->SetTileScale(s, s); }
  svtkSetVector4Macro(TileViewport, double);
  svtkGetVector4Macro(TileViewport, double);
  //@}

protected:
  svtkWindow();
  ~svtkWindow() override;

  char* WindowName;
  int Size[2];
  int Position[2];
  svtkTypeBool Mapped;
  bool ShowWindow;
  bool UseOffScreenBuffers;
  svtkTypeBool Erase;
  svtkTypeBool DoubleBuffer;
  int DPI;

  double TileViewport[4];
  int TileSize[2];
  int TileScale[2];

private:
  svtkWindow(const svtkWindow&) = delete;
  void operator=(const svtkWindow&) = delete;
};

#endif
