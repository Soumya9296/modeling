### We will try to model the interior of super earths with the codes that worked well for interior modeling of Earth and Mars

# SUPER EARTH: Earth 2.0 - also k/a Kepler-452b
# Radius = 1.63 R⊕   (NASA Exoplanet Catalogue)
# Mass = 3.29 M⊕     (NASA Exoplanet Catalogue)   [Mass = 3.78 +- 2.78 M⊕ (Bodog et al, 2023)]
# Orbital radius = 1.046 AU, Orbital period = 384.8 Earth Days



from __future__ import absolute_import
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import os
import sys  
sys.path.insert(0, 'S:/modeling/tools/burnman/burnman-2.1/burnman') 

import burnman

from burnman import Mineral, Material, Composite, Layer, Planet
from burnman import minerals
from burnman import CombinedMineral

from burnman import Composition
from burnman.tools.chemistry import formula_mass


import warnings




#### Approach 1: Rocky planet model with 3 layers i.e. core, mantle, and crust (Aligned with Jenkins et al. 2015; PMF IAS Earth layers)




if __name__ == "__main__":

    # FIRST: we must define the composition of the planet as individual layers.
    # A layer is defined by 4 parameters: Name, min_depth, max_depth, and number of slices within the layer.
    # Separately the composition and the temperature_mode need to set.

    # Radius of Kepler-452b = 10385.e3
    ratio = 10385/6371

    radius_planet = 6371.e3*ratio   # 10385 km
    core_radii = 3480.e3*ratio      # 5672.55 km
    mantle_radii = 6151.e3*ratio    # 10026.40 km
    litho_radii = 6346.6e3*ratio      # 10339.36 km


    # 1st Layer: Core

    core_composition = Composition({'Fe': 85.00, 'Ni': 14.00, 'S': 0.00, 'Si': 0.50, 'O': 0.25, 'C': 0.125, 'H': 0.125}, 'weight') # fig 2

    for c in [core_composition]:
        c.renormalize('atomic', 'total', 1.)

    core_elemental_composition = dict(core_composition.atomic_composition)
    core_molar_mass = formula_mass(core_elemental_composition)

    #    print(core_elemental_composition)
    #    print(core_molar_mass)

    core = Layer('core', radii=np.linspace(0, core_radii, 100))

    hcp_iron = minerals.SE_2015.hcp_iron()
    params = hcp_iron.params

    params['name'] = 'modified hcp iron'
    params['formula'] = core_elemental_composition
    params['molar_mass'] = core_molar_mass
    delta_V = 2.3e-7
    core_material = Mineral(params=params,
                                  property_modifiers=[['linear',
                                                       {'delta_E': 0.,
                                                        'delta_S': 0.,
                                                        'delta_V': delta_V}]])

    # check that the new inner core material does what we expect:
    # performing a validation check on the volume difference between
    # hcp_iron and inner_core_material at a specified pressure (40 GPa) and
    # temperature (2500 K)

    hcp_iron.set_state(800.e9, 7000.)
    core_material.set_state(800.e9, 7000.)
    assert np.abs(delta_V - (core_material.V - hcp_iron.V)) < 1.e-12

    # computed volume difference (inner_core_material.V - hcp_iron.V) is numerically
    # equal to delta_V within a very small tolerance (1.e-12)

    core.set_material(core_material)
    
    # core.set_material(burnman.minerals.other.Liquid_Fe_Anderson()) #s change material
    # The minerals that make up our core do not currently implement the thermal equation of state, so we will define the temperature at 300 K.
    
    core.set_temperature_mode("adiabatic", temperature_top = 5000.0) 



    # 2nd Layer: Mantle

    mantle = burnman.Layer("mantle", radii=np.linspace(core_radii, mantle_radii, 100))

    # High-pressure mantle phases (SLB_2024 database)
    ppv = minerals.SLB_2024.post_perovskite(molar_fractions=[0.80, 0.20, 0, 0, 0])          # Mg-Fe post-perovskite
    fer = minerals.SLB_2024.ferropericlase(molar_fractions=[0.70, 0.30, 0, 0, 0])           # Higher Fe content
    bdg = minerals.SLB_2024.bridgmanite(molar_fractions=[0.85, 0.10, 0.05, 0, 0, 0, 0])     # Bridgmanite (Mg-dominated)
    capv = minerals.SLB_2024.capv()                                                         # Calcium perovskite

    # Mantle rock composite (adjusted for super-Earth conditions)
    kepler_mantle = Composite([ppv, fer, bdg, capv], [0.45, 0.35, 0.15, 0.05]) #S
    mantle.set_material(kepler_mantle)
    

    # Here we add a thermal boundary layer perturbation, assuming that the
    # mantle has a Rayleigh number of 1.e7, and that there
    # is an 800 K jump across the basal thermal boundary layer and a
    # 100 K jump at the top of the mantle.

    tbl_perturbation = burnman.BoundaryLayerPerturbation(
        radius_bottom = core_radii, 
        radius_top = mantle_radii, 
        rayleigh_number = 1.0e7, 
        temperature_change = 800.0, 
        boundary_layer_ratio=100/800, 
    )

    mantle_radius = np.linspace(core_radii, mantle_radii, 100)

    dT_superadiabatic = 300.*(mantle_radius - mantle_radius[-1])/(mantle_radius[0] - mantle_radius[-1])
    mantle_tbl = (tbl_perturbation.temperature(mantle_radius) + dT_superadiabatic)
    mantle.set_temperature_mode('perturbed-adiabatic', temperatures=mantle_tbl)



    # 3rd Layer: Lithosphere

    lithosphere = burnman.Layer("lithosphere", radii = np.linspace(mantle_radii, litho_radii, 100))

    # Adjusted for high-pressure, high-gravity lithosphere (1.6–2× Earth gravity)
    kepler_litho_rock = Composite(
        [minerals.SLB_2024.olivine(molar_fractions=[0.70, 0.30]),                            # Fo₇₀Fa₃₀ (Fe-enriched for high-pressure stability)
        minerals.SLB_2024.orthopyroxene(molar_fractions=[0.80, 0.20, 0.0, 0.0]),             # En₈₀Fs₂₀  
        #minerals.SLB_2024.majorite(molar_fractions=[0.75, 0.15, 0.10, 0.0]),                # Mg-rich majorite (high-pressure garnet)
        minerals.SLB_2024.mgmj(),
        minerals.SLB_2024.clinopyroxene(molar_fractions=[0.75, 0.25, 0.0, 0.0, 0.0, 0.0])],  # Di₇₅Hd₂₅  
        [0.50, 0.25, 0.10, 0.15]                                                             # Volume fractions  
    )
    kepler_litho_rock.set_method('slb3')  # Stixrude & Lithgow-Bertelloni EoS

    # Assign to lithosphere layer
    lithosphere.set_material(kepler_litho_rock)

    lithosphere.set_temperature_mode("adiabatic", 
                                     #temperature_top = 500
                                     ) #S

    # ######## WORKING 

    # 4th Layer: Crust

    crust = Layer("crust", radii=np.linspace(litho_radii, radius_planet, 45))
    

    # Crust composition (basaltic, no Fe-rich minerals)  
    kepler_crust_rock = Composite(  
        [minerals.SLB_2024.plagioclase(molar_fractions=[0.65, 0.35]),             # An₆₅Ab₃₅ plagioclase  
        minerals.SLB_2024.diopside(),                                             # Mg-rich clinopyroxene  
        minerals.SLB_2024.quartz()],                                              # SiO₂ (coesite/stishovite at depth)  
        [0.60, 0.30, 0.10]                                                        # Volume fractions  
    )  
    kepler_crust_rock.set_method('slb3')  

    crust.set_material(kepler_crust_rock) 

    crust.set_temperature_mode("user-defined", np.linspace(400, 300, 45)) #S
    
        
    
    # Now we calculate the planet
        
    planet_se = Planet("Planet Kepler-452b", [core, mantle, lithosphere, crust], verbose=True)
    print(planet_se)

    # Here we compute its state. Go BurnMan Go!
    # (If we were to change composition of one of the layers, we would have to
    # recompute the state)

    planet_se.make()

    # Now we output the mass of the planet and moment of inertia

    print("n\mass of the model planet = {0:.5e}".format(planet_se.mass))

    print(
        "\nmass/Earth= {0:.5f}, moment of inertia factor= {1:.6f}".format(
            planet_se.mass/ 5.972e24, planet_se.moment_of_inertia_factor
        )
    )

    # And here's the mass of the individual layers:
    for layer in planet_se.layers:
        print(
            "{0} mass fraction of planet {1:.4f}".format(
                layer.name, layer.mass / planet_se.mass
            )
        )
    print("")


   # No reference model is available, so there is no need to compare with a reference seismic model


    # Now let's plot everything up

    # Optional prettier plotting

    plt.style.use('ggplot')

    figure = plt.figure(figsize=(12, 7))
    figure.suptitle(
        "{0} has a mass {1:.5f} times that of Earth,\n"
        "has an average density of {2:.3f} kg/m$^3$,\n"
        "and a moment of inertia factor of {3:.6f}".format(
            planet_se.name,
            planet_se.mass / 5.972e24,
            planet_se.average_density,
            planet_se.moment_of_inertia_factor,
        ),
        fontsize=12,
    )

    ax = [figure.add_subplot(2, 2, i) for i in range(1, 5)]

    bounds = np.array([[layer.radii[0]/1.e3, layer.radii[-1]/1.e3]
                   for layer in planet_se.layers])

    # maxy = [8, 50, 6, 4000]  #s setting maximum values of Y axis

    layer_colors = ['#FF9A7F', '#FF99FF', '#99FF99', '#8080C0']


    for bound, color in zip(bounds, layer_colors):
        for i in range(4):
            ax[i].axvspan(bound[0], bound[1], facecolor=color, alpha=0.35)


    ax[0].plot(planet_se.radii / 1.0e3, planet_se.density / 1.0e3, "k", linewidth=0.75, label=planet_se.name)
    #ax[0].plot(marskbradii / 1.0e3, marskbdensity / 1.0e3, "--k", linewidth=0.5, label="MARSKB")
    ax[0].set_ylim(0.0, (max(planet_se.density) / 1.0e3) + 5.0)
    #ax[0].set_ylim(0.0, 10.0)
    ax[0].set_xlim(0.0, 10400.0)
    ax[0].set_ylabel("Density ($\\cdot 10^3$ kg/m$^3$)")
    ax[0].legend()

    # Make a subplot showing the calculated pressure profile
    ax[1].plot(planet_se.radii / 1.0e3, planet_se.pressure / 1.0e9, "b", linewidth=0.75, label=planet_se.name)
    #ax[1].plot(marskbradii / 1.0e3, marskbpressure, "--b", linewidth=0.5, label="MARSKB")
    ax[1].set_ylim(0.0, (max(planet_se.pressure) / 1e9) + 500.0)
    ax[1].set_xlim(0.0, 10400.0)
    #ax[1].set_ylim(0.0, 500.0)
    ax[1].set_ylabel("Pressure (GPa)")

    # Make a subplot showing the calculated gravity profile
    ax[2].plot(planet_se.radii / 1.0e3, planet_se.gravity, "g", linewidth=0.75)
    #ax[2].plot(marskbradii / 1.0e3, marskbgravity, "--g", linewidth=0.5)
    ax[2].set_ylim(0.0, max(planet_se.gravity) + 5.0)
    ax[2].set_xlim(0.0, 10400.0)
    #ax[2].set_ylim(0.0, 20.0)
    ax[2].set_ylabel("Gravity (m/s$^2)$")
    

    # Make a subplot showing the calculated temperature profile
    mask = planet_se.temperature > 100.0  #s
    ax[3].plot(planet_se.radii[mask] / 1.0e3, planet_se.temperature[mask], "r", linewidth=0.75)
    ax[3].set_ylabel("Temperature ($K$)")
    ax[3].set_xlabel("Radius (km)")
    ax[3].set_ylim(0.0, max(planet_se.temperature) + 2000)
    ax[3].set_xlim(0.0, 10400.0)
    #ax[3].set_ylim(0.0, 8000)

    for i in range(2):
        ax[i].set_xticklabels([])
    for i in range(4):
        ax[i].set_xlim(0.0, max(planet_se.radii) / 1.0e3)

    figure.set_tight_layout(True)
    plt.show()
