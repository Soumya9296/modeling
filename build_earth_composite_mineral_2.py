# Building Earth: composition of mantle is composite minerals and reference model PREM
 
import numpy as np
import matplotlib.pyplot as plt
import burnman
from burnman import Mineral, PerplexMaterial, Composite, Layer, Planet
from burnman import minerals

"""### The Planet class

In a 1D Planet, the pressure, gravity, temperature and temperature gradient at the interfaces between layers must be continuous. In BurnMan, it is possible to collect layers together into a Planet, and have the "make" method of Planet work out how to ensure continuity (at least for pressure, gravity and temperature; for the sake of flexibility, temperature gradient is allowed to be discontinuous).

In the following example, we build Model Earth, a planet similar to Earth but a little simpler in mineralogical makeup. First, we create an adiabatic inner core. The inner core probably isn't adiabatic, but this is largely unimportant for the innermost layer.
"""

# Assuming 6 layer planet: **Model Earth** - inner core, outer core, lower mantle, upper mantle, lithosphere, and crust.
# Boundaries between the layers are inner core boundary (icb), core mantle boundary (cmb), lower mantle-upper mantle boundary, lithosphere-asthenoshpere boundary (lab), upper mantle-crust boundary (moho) respectively.

# 1st layer: Inner core


from burnman import Composition
from burnman.tools.chemistry import formula_mass

# Compositions from midpoints of Hirose et al. (2021), ignoring carbon and hydrogen

#run6.1 - 6 in PPT

inner_core_composition = Composition({'Fe': 94.485, 'Ni': 5., 'S': 0.10, 'Si': 0.30, 'O': 0.10, 'C': 0.01, 'H': 0.005}, 'weight')
outer_core_composition = Composition({'Fe': 91.26, 'Ni': 5., 'S': 1.70, 'Si': 0.95, 'O': 0.80, 'C': 0.20, 'H': 0.09}, 'weight')

for c in [inner_core_composition, outer_core_composition]:
    c.renormalize('atomic', 'total', 1.)

inner_core_elemental_composition = dict(inner_core_composition.atomic_composition)
outer_core_elemental_composition = dict(outer_core_composition.atomic_composition)
inner_core_molar_mass = formula_mass(inner_core_elemental_composition)
outer_core_molar_mass = formula_mass(outer_core_elemental_composition)

print(inner_core_elemental_composition)
print(inner_core_molar_mass)

icb_radius = 1220.e3
inner_core = Layer('inner core', radii=np.linspace(0., icb_radius, 21))

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

inner_core.set_temperature_mode('adiabatic')

"""Now, we create an adiabatic outer core."""

# 2nd layer: Outer core

cmb_radius = 3480.e3
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

# check that the new inner core material does what we expect:
# performing a validation check on the volume difference between
# hcp_iron and inner_core_material at a specified pressure (200 GPa) and
# temperature (4000 K)

liq_iron.set_state(200.e9, 4000.)
outer_core_material.set_state(200.e9, 4000.)
assert np.abs(delta_V - (outer_core_material.V - liq_iron.V)) < 1.e-12

# computed volume difference (inner_core_material.V - hcp_iron.V) is numerically
# equal to delta_V within a very small tolerance (1.e-12)

outer_core.set_material(outer_core_material)

outer_core.set_temperature_mode('adiabatic')

# 3rd layer: Lower mantle

# Let's apply a perturbed adiabatic temperature profile.

from burnman import BoundaryLayerPerturbation

lower_mantle_radius = 5701.e3 # 2221 km thick lower mantle
lower_mantle_temperature = 1873. #s value at 670 km depth - taken from geotherm (Brown, 1981)

convecting_lower_mantle_radii = np.linspace(cmb_radius, lower_mantle_radius, 2221)
convecting_lower_mantle = Layer('convecting lower mantle', radii=convecting_lower_mantle_radii)


# LM_rock = burnman.Composite([burnman.minerals.SLB_2024.bridgmanite(), burnman.minerals.SLB_2024.ferropericlase(), burnman.minerals.SLB_2024.capv()], [0.84, 0.09, 0.07])
# LM_rock = Composite([minerals.SLB_2024.mgpv(), minerals.SLB_2024.pe(), minerals.SLB_2024.capv()], [0.84, 0.09, 0.07])

bdg = minerals.SLB_2024.bridgmanite(molar_fractions=[0.85, 0.10, 0.05, 0, 0, 0, 0])
fer = minerals.SLB_2024.ferropericlase(molar_fractions=[0.80, 0.20, 0, 0, 0])
capv = minerals.SLB_2024.capv()
LM_rock = Composite([bdg, fer, capv], [0.84, 0.09, 0.07])
convecting_lower_mantle.set_material(LM_rock)


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
convecting_lower_mantle.set_temperature_mode('perturbed-adiabatic',
                                       temperatures=convecting_lower_mantle_tbl)

# 4th layer: Upper Mantle

lab_radius = 6151.e3 # 220 km thick lithosphere as pe PREM (lab = lithosphere astheosphere boundary)
lab_temperature = 1548. #s Anderson, 1982

convecting_upper_mantle_radii = np.linspace(lower_mantle_radius, lab_radius, 220)
convecting_upper_mantle = Layer('convecting upper mantle', radii=convecting_upper_mantle_radii)

# UM_rock = burnman.Composite([burnman.minerals.SLB_2024.olivine(), burnman.minerals.SLB_2024.orthopyroxene(), burnman.minerals.SLB_2024.clinopyroxene(), burnman.minerals.SLB_2024.mg_fe_aluminous_spinel(), burnman.minerals.SLB_2024.garnet()], [0.55, 0.25, 0.10, 0.05, 0.05])
# UM_rock = Composite([minerals.SLB_2024.fo(), minerals.SLB_2024.en(), minerals.SLB_2024.di(), minerals.SLB_2024.sp(), minerals.SLB_2024.py()], [0.55, 0.25, 0.10, 0.05, 0.05])

ol = minerals.SLB_2024.olivine(molar_fractions = [0.90, 0.10])
opx = minerals.SLB_2024.orthopyroxene(molar_fractions = [0.85, 0.15, 0, 0])
cpx = minerals.SLB_2024.clinopyroxene(molar_fractions = [0.80, 0.00, 0.05, 0.15, 0, 0])
spinel = minerals.SLB_2024.mg_fe_aluminous_spinel(molar_fractions = [0.80, 0.10, 0.05, 0.05])
garnet = minerals.SLB_2024.garnet(molar_fractions = [0.85, 0.10, 0.05, 0, 0, 0, 0])

UM_rock = Composite([ol, opx, cpx, spinel, garnet], [0.55, 0.25, 0.10, 0.05, 0.05])

convecting_upper_mantle.set_material(UM_rock)

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
convecting_upper_mantle.set_temperature_mode('perturbed-adiabatic',
                                       temperatures=convecting_upper_mantle_tbl)

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
d = np.loadtxt('S:\modeling\perplex_data\Anzellini_2013_geotherm.dat')
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

#layer_colors = ['#a7c7e7', '#f7c6a4', '#b0dab9', '#e7b3b3', '#d4a3ff', '#0039a6']
#layer_colors = ['#C0C0C0', '#FF5733', '#FF00FF', '#33FF33', '#FFFF00', '#000080']
layer_colors = ['#E0E0E0', '#FF9A7F', '#FF99FF', '#99FF99', '#FFFFB3', '#8080C0']


for bound, color in zip(bounds, layer_colors):
    for i in range(4):
        ax[i].axvspan(bound[0], bound[1], facecolor=color, alpha=0.4)

#for bound in bounds:
#    for i in range(4):
#        ax[i].fill_betweenx([0., maxy[i]],
#                            [bound[0], bound[0]],
#                            [bound[1], bound[1]], alpha=0.2)

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