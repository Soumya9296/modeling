# -*- coding: utf-8 -*-
"""build_model_earth_1.0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Hgizw3rz-3jAutUZdyq0hftHZzqSgAjS

Mounting Google Dive to Google Colab
"""

#from google.colab import drive
#drive.mount('/content/drive')

"""Installing Burnman in cloud environment"""

#pip install burnman

"""Installing necessary libraries"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import burnman

"""At first, setting up inner and outer core along with it's materials.
Setting up inner and outer core composition from Hirose, 2021
"""

from burnman import Composition
from burnman.tools.chemistry import formula_mass

# Define compositions
inner_core_composition = Composition({'Fe': 92.44, 'Ni': 5., 'S': 0.55, 'Si': 1.15, 'O': 0.1, 'C': 0.65, 'H': 0.12}, 'weight')
outer_core_composition = Composition({'Fe': 88.07, 'Ni': 5., 'S': 1.70, 'Si': 2., 'O': 2.90, 'C': 0.20, 'H': 0.13}, 'weight')

# Define compositions (taking lower bound from Hirose et al., 2021)
#inner_core_composition = Composition({'Fe': 93.40, 'Ni': 5., 'S': 0.15, 'Si': 1.00, 'O': 0.10, 'C': 0.25, 'H': 0.10}, 'weight')
#outer_core_composition = Composition({'Fe': 91.20, 'Ni': 5., 'S': 1.70, 'Si': 1., 'O': 0.80, 'C': 0.20, 'H': 0.10}, 'weight')


# Renormalize compositions
for c in [inner_core_composition, outer_core_composition]:
    c.renormalize('atomic', 'total', 1.)

# Convert to dictionaries with rounded values
inner_core_elemental_composition = {k: round(v, 4) for k, v in inner_core_composition.atomic_composition.items()}
outer_core_elemental_composition = {k: round(v, 4) for k, v in outer_core_composition.atomic_composition.items()}

# Compute molar masses
inner_core_molar_mass = formula_mass(inner_core_elemental_composition)
outer_core_molar_mass = formula_mass(outer_core_elemental_composition)

# Print results
print(inner_core_elemental_composition)
print(f'Molar Mass of inner core is: {inner_core_molar_mass:.4f} kg/mol')

print(outer_core_elemental_composition)
print(f'Molar Mass of outer core is: {outer_core_molar_mass:.4f} kg/mol')

"""Assuming **6 layer planet**: **Model Earth** - inner core, outer core, lower mantle, upper mantle, lithosphere, and crust.

Boundaries between the layers are inner core boundary (icb), core mantle boundary (cmb), lower mantle-upper mantle boundary, lithosphere-asthenoshpere boundary (lab), upper mantle-crust boundary (moho) respectively.

**1st layer:** Inner core
"""

from burnman import Mineral, PerplexMaterial, Composite, Layer, Planet
from burnman import minerals

icb_radius = 1221.5e3  #icb = inner core boundary (a/c PREM)
inner_core = Layer('inner core', radii=np.linspace(0., icb_radius, 100))

hcp_iron = minerals.SE_2015.hcp_iron()
params = hcp_iron.params

params['name'] = 'modified solid iron'
params['formula'] = inner_core_elemental_composition
params['molar_mass'] = inner_core_molar_mass
delta_V = 2.0e-7

inner_core_material = Mineral(params=params,
                              property_modifiers=[['linear',
                                                   {'delta_E': 0.,
                                                    'delta_S': 0.,
                                                    'delta_V': delta_V}]])

# check that the new inner core material does what we expect:
# performing a validation check on the volume difference between
# hcp_iron and inner_core_material at a specified pressure (200 GPa) and
# temperature (4000 K)

hcp_iron.set_state(200.e9, 4000.)
inner_core_material.set_state(200.e9, 4000.)
assert np.abs(delta_V - (inner_core_material.V - hcp_iron.V)) < 1.e-12

# computed volume difference (inner_core_material.V - hcp_iron.V) is numerically
# equal to delta_V within a very small tolerance (1.e-12)

inner_core.set_material(inner_core_material)

inner_core.set_temperature_mode('adiabatic') #s for inner core temp mode does not matter

"""**2nd layer:** Outer core (adiabatic)"""

cmb_radius = 3480.e3     #s cmb = core mantle boundary
outer_core = Layer('outer core', radii=np.linspace(icb_radius, cmb_radius, 21))

liq_iron = minerals.SE_2015.liquid_iron()
params = liq_iron.params

params['name'] = 'modified liquid iron'
params['formula'] = outer_core_elemental_composition
params['molar_mass'] = outer_core_molar_mass
delta_V = -2.3e-7
outer_core_material = Mineral(params=params,
                              property_modifiers=[['linear',
                                                   {'delta_E': 0.,
                                                    'delta_S': 0.,
                                                    'delta_V': delta_V}]])

# check that the new outer core material does what we expect:
liq_iron.set_state(200.e9, 4000.)
outer_core_material.set_state(200.e9, 4000.)
assert np.abs(delta_V - (outer_core_material.V - liq_iron.V)) < 1.e-12

outer_core.set_material(outer_core_material)

outer_core.set_temperature_mode('adiabatic')

"""Now, let's assume that there are duoble mantle layers that are convecting. Import a PerpleX input table that contains the material properties of pyrolite for this two layer. There are two reasons why we use a PerpleX input table, rather than a native BurnMan composite:


1.   To allow the planet to have a thermodynamically equilibrated mantle (i.e., one where all of the minerals are stable at the given pressure and temperature). While BurnMan can be used to equilibrate known assemblages using the burnman.equilibrate() function, Python is insufficiently fast to calculate equilibrium assemblages on-the-fly.

2.   To illustrate the coupling with PerpleX.

Let's apply a perturbed adiabatic temperature profile.

**3rd layer:** Lower Mantle
"""

from burnman import BoundaryLayerPerturbation

lower_mantle_radius = 5701.e3 # 2221 km thick lower mantle
lower_mantle_temperature = 1873. #s value at 670 km depth - taken from geotherm (Brown, 1981)

convecting_lower_mantle_radii = np.linspace(cmb_radius, lower_mantle_radius, 2221)
convecting_lower_mantle = Layer('convecting lower mantle', radii=convecting_lower_mantle_radii)

# Import a low resolution PerpleX data table.
fname = 'S:\data\pyrolite_perplex_table_lo_res.dat'
pyrolite = PerplexMaterial(fname, name='pyrolite')
convecting_lower_mantle.set_material(pyrolite)

# Here we add a thermal boundary layer perturbation, assuming that the
# lower mantle has a Rayleigh number of 1.e7, and that the basal thermal
# boundary layer has a temperature jump of 840 K and the top
# boundary layer has a temperature jump of 100 K. (100 & 840 K)
tbl_perturbation = BoundaryLayerPerturbation(radius_bottom=cmb_radius,
                                             radius_top=lower_mantle_radius,
                                             rayleigh_number=1.e7,
                                             temperature_change=940.,
                                             boundary_layer_ratio=100./940.)

# Onto this perturbation, we add a linear superadiabaticity term according
# to Anderson (he settled on 200 K over the lower mantle)
dT_superadiabatic = 300.*(convecting_lower_mantle_radii - convecting_lower_mantle_radii[-1])/(convecting_lower_mantle_radii[0] - convecting_lower_mantle_radii[-1])

convecting_lower_mantle_tbl = (tbl_perturbation.temperature(convecting_lower_mantle_radii)
                         + dT_superadiabatic)

# Clip the temperatures to the allowed range for the material
#convecting_lower_mantle_tbl = np.clip(convecting_lower_mantle_tbl, 0.0, 4200.0)

convecting_lower_mantle.set_temperature_mode('perturbed-adiabatic',
                                       temperatures=convecting_lower_mantle_tbl)

"""**4th layer:** Upper Mantle"""

lab_radius = 6151.e3 # 220 km thick lithosphere as pe PREM (lab = lithosphere astheosphere boundary)
lab_temperature = 1548. #s Anderson, 1982

convecting_upper_mantle_radii = np.linspace(lower_mantle_radius, lab_radius, 220)
convecting_upper_mantle = Layer('convecting upper mantle', radii=convecting_upper_mantle_radii)

# Import a low resolution PerpleX data table.
fname1 = 'S:\data\pyrolite_perplex_table_lo_res.dat'
pyrolite1 = PerplexMaterial(fname1, name='pyrolite1')
convecting_upper_mantle.set_material(pyrolite1)

# Here we add a thermal boundary layer perturbation, assuming that the
# lower mantle has a Rayleigh number of 1.e7, and that the basal thermal
# boundary layer has a temperature jump of 100 K and the top
# boundary layer has a temperature jump of 60 K. (60 & 100 K)
tbl_perturbation1 = BoundaryLayerPerturbation(radius_bottom=lower_mantle_radius,
                                             radius_top=lab_radius,
                                             rayleigh_number=1.e7,
                                             temperature_change=160.,
                                             boundary_layer_ratio=60./160.)

# Onto this perturbation, we add a linear superadiabaticity term according
# to Anderson (he settled on 200 K over the lower mantle)
dT_superadiabatic1 = 300.*(convecting_upper_mantle_radii - convecting_upper_mantle_radii[-1])/(convecting_upper_mantle_radii[0] - convecting_upper_mantle_radii[-1])

convecting_upper_mantle_tbl = (tbl_perturbation1.temperature(convecting_upper_mantle_radii)
                         + dT_superadiabatic1)

# Clip the temperatures to the allowed range for the material
#convecting_upper_mantle_tbl = np.clip(convecting_upper_mantle_tbl, 0.0, 4200.0)

convecting_upper_mantle.set_temperature_mode('perturbed-adiabatic',
                                       temperatures=convecting_upper_mantle_tbl)

print(convecting_upper_mantle_tbl)

print("Min Superadiabatic Contribution:", np.min(dT_superadiabatic1))
print("Max Superadiabatic Contribution:", np.max(dT_superadiabatic1))

print(f"Upper Mantle Min: {np.min(convecting_upper_mantle_tbl)}, Max: {np.max(convecting_upper_mantle_tbl)} | "
      f"Lower Mantle Min: {np.min(convecting_lower_mantle_tbl)}, Max: {np.max(convecting_lower_mantle_tbl)}")

"""**5th layer:** Lithosphere - The lithosphere has a user-defined conductive gradient."""

moho_radius = 6346.e3 #s moho radius as per PREM
moho_temperature = 614. #s interpolated from Anderson, 1982

dunite = minerals.SLB_2011.mg_fe_olivine(molar_fractions=[0.92, 0.08])
lithospheric_mantle = Layer('lithospheric mantle',
                            radii=np.linspace(lab_radius, moho_radius, 195))
lithospheric_mantle.set_material(dunite)
lithospheric_mantle.set_temperature_mode('user-defined',
                                         np.linspace(lab_temperature,
                                                     moho_temperature, 195))

"""**6th layer:** Crust

Finally, we assume the crust has the density of andesine ~ 40% anorthite
"""

planet_radius = 6371.e3  #s 25 km thick crust
surface_temperature = 300.
andesine = minerals.SLB_2011.plagioclase(molar_fractions=[0.4, 0.6]) #s 40% Anorthite and 60% Albite
crust = Layer('crust', radii=np.linspace(moho_radius, planet_radius, 25))
crust.set_material(andesine)
crust.set_temperature_mode('user-defined',
                           np.linspace(moho_temperature,
                                       surface_temperature, 25))

"""Let's make our planet from its consistuent 6 layers."""

model_earth = Planet('Model Earth',
                    [inner_core, outer_core,
                     convecting_lower_mantle, convecting_upper_mantle, lithospheric_mantle,
                     crust], verbose=True)
model_earth.make()

"""Printing outputs: Total Mass, Moment of Inertia, Mass of Individual layers, and mass fractions"""

ref_earth_mass = 5.972e24
ref_earth_moment_of_inertia_factor = 0.3307

print(f'mass = {model_earth.mass:.3e} (Reference Earth = {ref_earth_mass:.3e})')
print(f'moment of inertia factor= {model_earth.moment_of_inertia_factor:.4f} '
      f'(Earth = {ref_earth_moment_of_inertia_factor:.4f})')

print('Layer mass fractions:')
for layer in model_earth.layers:
    print(f'{layer.name}: {layer.mass / model_earth.mass:.3f}')

from burnman.tools.output_seismo import write_axisem_input
from burnman.tools.output_seismo import write_mineos_input

write_axisem_input([convecting_lower_mantle, convecting_upper_mantle, lithospheric_mantle, crust], modelname='earth_silicates', plotting=True)
write_mineos_input([convecting_lower_mantle, convecting_upper_mantle, lithospheric_mantle, crust], modelname='earth_silicates', plotting=True)

# Now we delete the newly-created files. If you want them, comment out these lines.
import os
os.remove('axisem_earth_silicates.txt')
os.remove('mineos_earth_silicates.txt')

import warnings
prem = burnman.seismic.PREM()
premdepth = prem.internal_depth_list()
premradii = 6371.e3 - premdepth

with warnings.catch_warnings(record=True) as w:
    eval = prem.evaluate(['density', 'pressure', 'gravity', 'v_s', 'v_p'])
    premdensity, prempressure, premgravity, premvs, premvp = eval
    print(w[-1].message)

"""Creating the Anzellini et al. (2013) geotherm:"""

from scipy.interpolate import interp1d
d = np.loadtxt('S:\data\Anzellini_2013_geotherm.dat')
Anz_interp = interp1d(d[:,0]*1.e9, d[:,1])

"""Plotting the outputs"""

#plt.style.use('ggplot')
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"  # Change to 'sans-serif', 'monospace', etc.
plt.rcParams["font.size"] = 9

fig = plt.figure(figsize=(8, 5))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]


bounds = np.array([[layer.radii[0]/1.e3, layer.radii[-1]/1.e3]
                   for layer in model_earth.layers])

maxy = [15, 400, 12, 7000]  #s setting maximum values of Y axis
for bound in bounds:
    for i in range(4):
        ax[i].fill_betweenx([0., maxy[i]],
                            [bound[0], bound[0]],
                            [bound[1], bound[1]], alpha=0.2)

ax[0].plot(model_earth.radii / 1.e3, model_earth.density / 1.e3, c='#5405ad', linestyle='-',
           label=model_earth.name, linewidth=1.5)
ax[0].plot(premradii / 1.e3, premdensity / 1.e3, linestyle=':', c='black', label='PREM', linewidth=1.5)
ax[0].set_ylabel('Density ($10^3$ kg/m$^3$)')
ax[0].legend()

# Make a subplot showing the calculated pressure profile
ax[1].plot(model_earth.radii / 1.e3, model_earth.pressure / 1.e9, c='#5405ad')
ax[1].plot(premradii / 1.e3, prempressure / 1.e9, c='black', linestyle=':')
ax[1].set_ylabel('Pressure (GPa)')

# Make a subplot showing the calculated gravity profile
ax[2].plot(model_earth.radii / 1.e3, model_earth.gravity, c='#5405ad')
ax[2].plot(premradii / 1.e3, premgravity, c='black', linestyle=':')
ax[2].set_ylabel('Gravity (m/s$^2)$')
ax[2].set_xlabel('Radius (km)')

# Make a subplot showing the calculated temperature profile
ax[3].plot(model_earth.radii / 1.e3, model_earth.temperature, c='#5405ad')
ax[3].set_ylabel('Temperature (K)')
ax[3].set_xlabel('Radius (km)')
ax[3].set_ylim(0.,)

# Finally, let's overlay some geotherms onto our model
# geotherm
labels = ['Stacey (1977)',
          'Brown and Shankland (1981)',
          'Anderson (1982)',
          'Alfe et al. (2007)',
          'Anzellini et al. (2013)']

short_labels = ['S1977',
                'BS1981',
                'A1982',
                'A2007',
                'A2013']

ax[3].plot(model_earth.radii / 1.e3,
burnman.geotherm.stacey_continental(model_earth.depth),
linestyle='--', label=short_labels[0], linewidth=0.9)
mask = model_earth.depth > 269999.
ax[3].plot(model_earth.radii[mask] / 1.e3,
           burnman.geotherm.brown_shankland(model_earth.depth[mask]),
           linestyle='--', label=short_labels[1], linewidth=0.9)
ax[3].plot(model_earth.radii / 1.e3,
           burnman.geotherm.anderson(model_earth.depth),
           linestyle='--', label=short_labels[2], linewidth=0.9)

ax[3].scatter([model_earth.layers[0].radii[-1] / 1.e3,
               model_earth.layers[1].radii[-1] / 1.e3],
              [5400., 4000.],
              linestyle='--', label=short_labels[3], linewidth=0.9)

mask = model_earth.pressure < 330.e9
temperatures = Anz_interp(model_earth.pressure[mask])
ax[3].plot(model_earth.radii[mask] / 1.e3, temperatures,
           linestyle='--', label=short_labels[4], linewidth=0.9)

ax[3].legend()

for i in range(2):
    ax[i].set_xticklabels([])
for i in range(4):
    ax[i].set_xlim(0., max(model_earth.radii) / 1.e3)
    ax[i].set_ylim(0., maxy[i])

fig.set_tight_layout(True)
plt.show()

#print(model_earth.temperature)

#for temp in model_earth.temperature:
#    print(temp)