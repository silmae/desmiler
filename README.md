# desmiler

Detecting and correcting smile aberration of a push-broom hyperspectral imager. 

## Usage

See test_main.py for usage examples.

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



