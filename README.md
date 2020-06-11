# desmiler

Fot detecting and correcting smile aberration of a push-broom hyperspectral imager. 
Contains various inspection tools to visualize the result of desmiling. 

## Notation and data format

A frame is output of one exposure of the imager. 
A cube is hyperspectral data cube consisting of consecutive frames along a scan.

Frames and cubes are expected to be in netcdf format for saving and loading. 
We use xarray Dataset and DataArray to store and manipulate the data.

Be prepared to fix bugs if your data is in different form than ours. 
Especially the dimension names of the xarray Datasets are mostly hard-coded. 
We use ('index', 'y', 'x') order where 'index' is the direction of the scan, 
'y' is dimension perpendicular to scan direction,  and 'x' is the spectral dimension. 
Many operations use underlaying numpy arrays, so different order will most probably 
break things.

## Outline

* smile_cerrection.py: Does all heavy lifting for desmiling.
* scan.py: For desmiling the result of a single scanning session. 
	See the file documentation for details.
* frame_inspector.py: Inspect frames visually with matplotlib plots. 
	Inspected frame can be overlayed with circle fit and line fit data. 
	Band pass can also be plotted.
* cube_inspector.py: Interactive cube inspection tool. See more below.
* test_main.py: Usage exampels.
* spectral_line.py: SpectralLine objects represent a single spectral line 
	of a frame.
* curve_fit.py: Optimization for circle and line fits used by SpectralLine.

## CubeInspector class

Interactive tool for inspecting and comparing original and desmiled hyperspectral 
image cubes. Three inspection modes: reflectance images, false color images, and 
spectral angle maps (SAM). 

### Key bindings

* Number row 1: select mode 1 (reflectance)
* Number row 2: select mode 2 (false color)
* Number row 3: select mode 3 (SAM)

Mode specific bindings

* a/d (mode 3): move spectral filter to left and right
* r (mode 3): toggle between radians and dot product

### Mouse bindings

* left click (modes 1 and 2): 
	select from wchich pixel the spectra is shown or 
	select band if clicked over the spectra plot
* shit + left click (any mode): 
	set first corner of SAM window, second time sets the other 
	corner and SAM is shown when mode 3 is selected. Other actions 
	after giving the first corner will unset it.
* alt: 
	while pressed, all actions are ignored, so that matplolib's 
	own commands (such as zooming) can be used.

## Create conda environment

Create environment with conda running

```conda env create -n smile --file smile_env.yml```

in conda prompt.

Change ```create``` to ```update``` when updating.

