====================
 General
====================

JoyMEX is a MEX-file which allows querying (multiple) Gamepad devices from MATLAB in Microsoft Windows.

A Pre-built binary for MATLAB R2009b (both 32-bit and 64-bit version) can be found in the MATLAB directory.

Also included is an example JoyTest.m which shows the usage of JoyMEX.

Note: JoyMEX uses Microsoft DirectInput to communicate to the devices, therefore Microsoft DirectX needs to
      installed on your system to be able to use JoyMEX.

====================
 Usage
====================

Before a device can be queried it must be initialized by calling:

 JoyMEX('init',deviceid)

Where deviceid is the device number (0-3). If you need to access more devices you should initialize them in separate calls.
After the initialization the MEX-file will typically be called in a loop. MATLAB\JoyTest.m shows an example of this.

When done with the devices they should be release by calling:
 
 clear JoyMEX

For more information type:

 help JoyMEX

====================
 Building
====================

Requirements:

1. MATLAB
2. Microsoft Visual C++ 2008
3. Microsoft DirectX SDK


Instructions:

1. Open JoyMEX_VS2008.sln in Microsoft Visual Studio 2008.
2. Configure your VC++ Directories ("Tools" -> "Options" -> "Projects and Solutions" -> "VC++ Directories"):

a. Make sure the following directories are on your "Library files"

- The Microsoft DirectX SDK Lib directory (should automatically have been configured by SDK installer).
- $MATLABROOT\extern\lib\$ARCH\microsoft
  (Where $MATLABROOT stands for the directory where MATLAB has been installed and
   $ARCH is either win32 or win64 depending for which "Platform" you are configuring).

b. Make sure the following directories are on your "Include files":

- The Microsoft DirectX SDK Include directory (should automatically have been configured by SDK installer).
- $MATLABROOT\extern\include (Where $MATLABROOT stands for the directory where MATLAB has been installed)

3. Choose your configuration(s) and Build the Solution. (The Release builds will be copied to the MATLAB directory).

