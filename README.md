# Term Structure Estimation
This is my attempt on implementing the term structure estimation methods 
as described in [[1]](#1). Furthermore, I have implemented two spline methods,
namely monotone cubic splines and monotone convex splines. While the implementation
of former method is provided by SciPy via its [PchipInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html),
I have attempted to implement the latter myself, as described in [[2]](#2) and [[3]](#3). Additionally,
for the spline methods, I have plotted both, the yield curve and the instantaneous forward curve.


## References
<a id="1">[1]</a> 
Schich, Sebastian. T. (1997). 
Estimating the German term structure. https://www.bundesbank.de/resource/blob/622394/ab816caf466eeea7dd2fef26d0fbfc30/mL/1997-10-01-dkp-04-data.pdf

<a id="1">[2]</a>
Hagan, Patrick S. and West, Graeme (2008).
Methods for Constructing a Yield Curve. https://downloads.dxfeed.com/specifications/dxLibOptions/HaganWest.pdf

<a id="1">[3]</a>
Dehlbom, Gustaf (2020).
Interpolation of the yield curve. http://uu.diva-portal.org/smash/get/diva2:1477828/FULLTEXT01.pdf