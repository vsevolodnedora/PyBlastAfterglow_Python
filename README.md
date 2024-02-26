# PyBlastAfterglow

Semi-analytic code to compute non-thermal, synchrotron radiation from ejecta. 

This is a _decommissioned_ version of the code written around 2021. 
It supports stratified, angular, and velocity-structured ejecta. Ejecta is discretized into _non-interacting_ elements. 
Dynamics is computed considering forward shock and energy-conserving blastwave evolution under thin-shell approximation.  
Radiation is computed via analytic approximants (broken power-law spectra). 

The code was tested and used in [GRB120717A rebrightening paper](https://iopscience.iop.org/article/10.3847/2041-8213/ac504a) and in [GRB170817 as a kilonova afterglow paper](https://academic.oup.com/mnras/article/506/4/5908/6329057).  

The code is released under the MIT license.  
Should you find it useful for your research, please consider citing: 

```latex
@article{Nedora:2022kjv,
    author = "Nedora, Vsevolod and Dietrich, Tim and Shibata, Masaru and Pohl, Martin and Menegazzi, Ludovica Crosato",
    title = "{Modeling kilonova afterglows: Effects of the thermal electron population and interaction with GRB outflows}",
    eprint = "2208.01558",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    doi = "10.1093/mnras/stad175",
    month = "8",
    year = "2022"
}
```

The new, C++ version of the code is available here [PyBlastAfterglowMag](https://github.com/vsevolodnedora/PyBlastAfterglowMag). 


