# desmiler

For detecting and correcting smile aberration of a push-broom hyperspectral imager. 
Contains various inspection tools to visualize the result of desmiling. 

## Notation and data format

A frame is output of one exposure of the imager. 
A cube is hyperspectral data cube consisting of consecutive frames along a scan.

Frames and cubes are expected to be in netcdf format for saving and loading. 
We use xarray Dataset and DataArray to store and manipulate the data.

## Imaging

1. Light reference (for aberration correction)
	* point imager towards fluorescence tube
	* call `ui.shoot_light()`
2. White reference
	* point imager towards white reference using selected scanning platform
	* call `ui.shoot_white()`
3. Dark reference
	* put the lens cap on
	* call `ui.shoot_dark()`
	* remove the lens cap
4. Set scanning parameters
	* point imager towards white reference target using selected scanning platform
	* call `ui.start_preview()` 
	* take note of the illuminated sensor area and write proper cropping to the control file
	* one can also adjust the focus of the imager at this stage
	* adjust exposure time as needed by calling `ui.exposure(<value>)`
	* set scanning speed and length to the control file (using arbitrary units, 
			e.g., mm/s and mm, respectively) 	
5. Run a scan
	* run scan by calling `ui.run_scan()`. One can set 
		`is_mock_scan = 1` in control file to just print the scanning parameters 
	* take note of the actual scanning time, which is printed to the console and 
		may differ from what was desired 
		if the imager cannot provide frames fast enough with selected exposure time
6. Inspect cube
	* call `ui.show_cube()` to start the CubeInspector. It will load the raw 
		cube if no reflectance cubes are found from the file system. 
	* adjust parameters and redo the scan if needed by repeating steps 4, 5, and 6
7. Set correction parameters
	* call `ui.show_light()`, which will show the light reference shot earlier. 
		Ignore badly fitted arcs for now.
	* select some well-separated emission lines and write their x-coordinates 
		to the control file
	* call `ui.show_light()` again. Fitted arcs should now lie along 
		the actual emission lines. If this is not the case, try adjusting the emission 
		line estimates or select more prominent lines. Bad fits are usually grossly out 
		of place, so they are easy to spot.
8. Make reflectance cube
	* call `ui.make_reflectance_cube()`, which will use the dark and white 
		references shot earlier
9. Run aberration correction
	* call `ui.make_desmiled_cube()`, which will show a visual representation 
		of the shift matrix used for the correction before starting. The shift matrix should 
		be relatively smooth and values should be relatively small (our values were approx. 
		0.1 % of the sensor's width). 
10. Check the results
	* call `ui.show_cube()`, which will now load the corrected reflectance cubes 
		alongside the uncorrected one


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
	select from which pixel the spectra are shown or 
	select band if clicked over the spectra plot
* shit + left click (any mode): 
	set first corner of SAM window, second time sets the other 
	corner and SAM is shown when mode 3 is selected. Other actions 
	after giving the first corner will unset it.
* alt: 
	while pressed, all actions are ignored, so that matplotlib's 
	own commands (such as zooming) can be used.

## Create conda environment

Create environment with conda running

```conda env create -n smile --file smile_env.yml```

in conda prompt.

Change ```create``` to ```update``` when updating.

## DOI

https://doi.org/10.5281/zenodo.3903218

