# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.


########### WORKING ON CODES ###########


"""
example_build_mars2
--------------------

For Earth we have well-constrained one-dimensional density models.  This allows us to
calculate pressure as a function of depth.  Furthermore, petrologic data and assumptions
regarding the convective state of the planet allow us to estimate the temperature.

For planets other than Earth we have much less information, and in particular we
know almost nothing about the pressure and temperature in the interior.  Instead, we tend
to have measurements of things like mass, radius, and moment-of-inertia.  We would like
to be able to make a model of the planet's interior that is consistent with those
measurements.

However, there is a difficulty with this.  In order to know the density of the planetary
material, we need to know the pressure and temperature.  In order to know the pressure,
we need to know the gravity profile.  And in order to the the gravity profile, we need
to know the density.  This is a nonlinear problem which requires us to iterate to find
a self-consistent solution.

This example allows the user to define layers of planets of known outer radius and self-
consistently solve for the density, pressure and gravity profiles. The calculation will
iterate until the difference between central pressure calculations are less than 1e-5.
The planet class in BurnMan (../burnman/planet.py) allows users to call multiple
properties of the model planet after calculations, such as the mass of an individual layer,
the total mass of the planet and the moment of inertia. See planets.py for information
on each of the parameters which can be called.

*Uses:*

* :doc:`mineral_database`
* :class:`burnman.Planet`
* :class:`burnman.Layer`

*Demonstrates:*

* setting up a planet
* computing its self-consistent state
* computing various parameters for the planet
* seismic comparison
"""

from __future__ import absolute_import
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import burnman

from burnman import Mineral, Material, Composite, Layer, Planet
from burnman import minerals
from burnman import CombinedMineral

from burnman import Composition
from burnman.tools.chemistry import formula_mass


import warnings


if __name__ == "__main__":

    # FIRST: we must define the composition of the planet as individual layers.
    # A layer is defined by 4 parameters: Name, min_depth, max_depth,and number of slices within the layer.
    # Separately the composition and the temperature_mode need to set.

    radius_planet = 3389.5e3 #s radius of mars = 3389.5 km (NASA Fact sheet and Kh22)

    # inner_core
    """
    inner_core = burnman.Layer("inner core", radii=np.linspace(0, 10, 10)) #s changed radius
    inner_core.set_material(burnman.minerals.other.Fe_Dewaele()) #s change material

    # The minerals that make up our core do not currently implement the thermal equation of state, so we will set the temperature at 300 K.
    inner_core.set_temperature_mode(
        "user-defined", 2100.0 * np.ones_like(inner_core.radii) #s 2100k ? from S19 in Kh22
    )
    """

    # 1st Layer: Core

    #core_composition = Composition({'Fe': 82.50, 'Ni': 3.47, 'S': 12.00, 'Si': 0.00, 'O': 2.00, 'C': 0.015, 'H': 0.015}, 'weight')

    #core_composition = Composition({'Fe': 82.50, 'Ni': 3.47, 'S': 12.00, 'Si': 0.00, 'O': 2.00, 'C': 0.015, 'H': 0.015}, 'weight') # fig 4
    #core_composition = Composition({'Fe': 81.00, 'Ni': 3.45, 'S': 12.50, 'Si': 0.00, 'O': 3.00, 'C': 0.025, 'H': 0.025}, 'weight') # fig 5
    core_composition = Composition({'Fe': 81.00, 'Ni': 3.45, 'S': 12.50, 'Si': 0.00, 'O': 3.00, 'C': 0.025, 'H': 0.025}, 'weight') # fig 6


    for c in [core_composition]:
        c.renormalize('atomic', 'total', 1.)

    core_elemental_composition = dict(core_composition.atomic_composition)
    core_molar_mass = formula_mass(core_elemental_composition)

    #    print(core_elemental_composition)
    #    print(core_molar_mass)

    #core_radius = 1650.e3  # from paper perplexity
    #core = Layer('core', radii=np.linspace(0, core_radius, 33))

    liq_iron = minerals.SE_2015.liquid_iron()
    params = liq_iron.params

    params['name'] = 'modified liquid iron'
    params['formula'] = core_elemental_composition
    params['molar_mass'] = core_molar_mass
    delta_V = -2.3e-7
    core_material = Mineral(params=params,
                                  property_modifiers=[['linear',
                                                       {'delta_E': 0.,
                                                        'delta_S': 0.,
                                                        'delta_V': delta_V}]])

    # check that the new inner core material does what we expect:
    # performing a validation check on the volume difference between
    # hcp_iron and inner_core_material at a specified pressure (40 GPa) and
    # temperature (2500 K)

    liq_iron.set_state(40.e9, 4000.)
    core_material.set_state(40.e9, 4000.)
    assert np.abs(delta_V - (core_material.V - liq_iron.V)) < 1.e-12

    # computed volume difference (inner_core_material.V - hcp_iron.V) is numerically
    # equal to delta_V within a very small tolerance (1.e-12)

    core = burnman.Layer("core", radii=np.linspace(0, 1791.645e3, 100)) #s changed radius from Kh22
    
    core.set_material(core_material)
    
    # core.set_material(burnman.minerals.other.Liquid_Fe_Anderson()) #s change material
    # The minerals that make up our core do not currently implement the thermal equation of state, so we will define the temperature at 300 K.
    
    core.set_temperature_mode("adiabatic", temperature_top = 2550.0) #s 2100 frm S19 in Kh22   2550

    """
    # 2nd Layer: Molten Silicate Layer

    msl_radii = np.linspace(1691.645e3, 1841.645e3, 150)
    msl = Layer("molten silicate layer", radii = msl_radii)

    silicate_melt = minerals.HGP_2018_ds633.silicate_melt(molar_fractions=[0.60, 0.00, 0.20, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])

    # Create composite with minor suspended crystals [4]

    fer = minerals.SLB_2024.ferropericlase(molar_fractions=[0.80, 0.20, 0, 0, 0])

    ###
    molten_layer = Composite([silicate_melt,               # 85% melt
                          minerals.SLB_2024.st(),          # 10% suspended stishovite crystals
                          fer],                            # 5% Fe-rich crystals 
                         [0.70, 0.10, 0.20])               # fig 4
    
    molten_layer = Composite([silicate_melt, minerals.SLB_2024.st(), fer], [0.60, 0.10, 0.30]) # fig 5
    ###
    #molten_layer.set_state(23.5e9, 2100)
    msl.set_material(molten_layer)
    msl.set_temperature_mode('adiabatic', temperature_top = 2240)

    """




    # 3rd Layer: Mantle

    mantle_radii = np.linspace(1791.645e3, 2830.303e3, 100) #S changed radius from Kh22
    mantle = burnman.Layer("mantle", radii=mantle_radii)
    #mantle.set_material(burnman.minerals.SLB_2011.mg_bridgmanite()) #s change material (bridgmanite is less likely)
    
    M_rock = Composite([
    minerals.SLB_2011.mg_fe_wadsleyite(molar_fractions=[0.75, 0.25]),           # Upper transition zone [Kiefer et al., 2015]
    minerals.SLB_2011.mg_fe_ringwoodite(molar_fractions=[0.75, 0.25]),          # Lower transition zone
    minerals.SLB_2011.garnet(molar_fractions=[0.00, 0.00, 0.00, 0.50, 0.50])    # Garnet component
    ], [0.50, 0.30, 0.20])
    
    mantle.set_material(M_rock)
    
    # Here we add a thermal boundary layer perturbation, assuming that the
    # lower mantle has a Rayleigh number of 1.e7, and that there
    # is an 800 K jump across the basal thermal boundary layer and a
    # 100 K jump at the top of the lower mantle.

    tbl_perturbation = burnman.BoundaryLayerPerturbation(
        radius_bottom=1791.645e3, #s changed (this is radius of core)
        radius_top=2830.303e3, #s changed (this is radius of mantle)
        rayleigh_number=1.0e6, #s changed as Mars's stagnant lid mantle (current thermal state), also try 10^6 to see a mixed state of localized convection and stagnation.
        temperature_change=470.0, #s changed a/c paper! cite (300-600 K)
        boundary_layer_ratio=100/470, #s changed as sluggish mantle with top-heavy thermal structure  # 0.0 means No boundary layer i.e. uniform perturbation
    )

#    mantle_tbl = tbl_perturbation.temperature(mantle_radii)
#    mantle.set_temperature_mode(
#        "perturbed-adiabatic", temperatures=mantle_tbl
#    )

    dT_superadiabatic = 300.*(mantle_radii - mantle_radii[-1])/(mantle_radii[0] - mantle_radii[-1])
    mantle_tbl = (tbl_perturbation.temperature(mantle_radii) + dT_superadiabatic)
    mantle.set_temperature_mode('perturbed-adiabatic', temperatures=mantle_tbl)




    # 4th Layer: Lithosphere

    lithosphere = burnman.Layer("lithosphere", radii = np.linspace(2830.303e3, 3346.303e3, 100)) ######## should be changed ?

    litho_rock = Composite(
    [minerals.SLB_2011.mg_fe_olivine(molar_fractions=[0.79, 0.21]),                    # Fo₇₉Fa₂₁ [Yoshizaki & McDonough, 2020]
     minerals.SLB_2011.orthopyroxene(molar_fractions=[0.85, 0.15, 0.00, 0.00]),        # En₈₅Fs₁₅ [Kiefer et al., 2015]
     minerals.SLB_2011.clinopyroxene(molar_fractions=[0.70, 0.30, 0.00, 0.00, 0.00]),  # Di₇₀Hd₃₀ [McSween et al., 2009]
     minerals.SLB_2011.spinel()],                                                      # MgAl₂O₄ [Ehlmann & Edwards, 2014]
    [0.55, 0.25, 0.15, 0.05]                                                           # Volume fractions [Khan et al., 2023]
    )
    litho_rock.set_method('slb3')  # Stixrude & Lithgow-Bertelloni EoS [2011]

    lithosphere.set_material(litho_rock)
    lithosphere.set_temperature_mode("user-defined", np.linspace(1650, 331, 100)) #S paper aereotherm




    # 5th Layer: Crust

    crust = burnman.Layer(
        "crust", radii=np.linspace(3346.303e3, 3389.5e3, 43) #s changed radius from Kh22
    )

    crust_rock = Composite(
    [minerals.SLB_2011.plagioclase(molar_fractions=[0.60, 0.40]),                           # An₆₀Ab₄₀ [Bandfield et al., 2000]
     minerals.SLB_2011.clinopyroxene(molar_fractions=[0.60, 0.00, 0.00, 0.00, 0.40]),       # Di₆₀Jd₄₀ [Wyatt et al., 2004]
     minerals.SLB_2024.mag()],                                                              # Fe₃O₄ [Ehlmann & Edwards, 2014]
    [0.60, 0.30, 0.10]                                                                      # Volume fractions [Samuel et al., 2023]
    )

    crust.set_material(crust_rock)
    
    #crust.set_material(burnman.minerals.SLB_2011.forsterite())

    #s creating array for giving temp values from Drilleau et al, 2021 -> Dri21 (I'll use three set for aereotherm)
    #crust_rads = np.array([3389.5, 3380.5, 3370.5, 3360.5, 3350.5, 3346.303])
    #crust_temps = np.array([220, 243.1285, 268.8268, 294.5251, 320.2235, 331.0091])
    #crust_temperatures = np.interp(crust.radii, crust_rads, crust_temps)
    # Set temperature mode to "user-defined" with the interpolated temperatures
    
    crust.set_temperature_mode("user-defined", np.linspace(331, 220, 43)) #S
    
    #crust.set_temperature_mode("adiabatic", temperature_top=220.0) #s 220K in S19 taken from Kh22

    """
    upper_mantle = burnman.Layer(
        "upper mantle", radii=np.linspace(3346.303e3, 3389.5e3, 10)
    )
    upper_mantle.set_material(burnman.minerals.SLB_2011.forsterite())
    upper_mantle.set_temperature_mode("adiabatic", temperature_top=1200.0) 

    """
    
    
    
    
    
    
    
    
    # Now we calculate the planet. #s name changed to mars
    
    
    planet_mars = burnman.Planet(
        "Planet Mars", [
            core, 
            #msl, 
            mantle, 
            lithosphere, 
            crust
            ], 
            verbose=True     
    )
    print(planet_mars)

    # Here we compute its state. Go BurnMan Go!
    # (If we were to change composition of one of the layers, we would have to
    # recompute the state)

    planet_mars.make()

    # Now we output the mass of the planet and moment of inertia

    print(
        "\nMass of Mars = {0:.5e}".format(
            planet_mars.mass
        )
    )

    print(
        "\nmass/Earth= {0:.5f}, moment of inertia factor= {1:.6f}".format(
            planet_mars.mass/ 5.972e24, planet_mars.moment_of_inertia_factor
        )
    )

    # And here's the mass of the individual layers:
    for layer in planet_mars.layers:
        print(
            "{0} mass fraction of planet {1:.4f}".format(
                layer.name, layer.mass / planet_mars.mass
            )
        )
    print("")

    
    """
    # Identify CMB index (end of mantle / beginning of core)
    cmb_index = max(i for i, name in enumerate(planet_mars.layers) if name == "mantle")

    # Get boundary values at CMB
    cmb_radius = planet_mars.radius[cmb_index]
    cmb_pressure = planet_mars.pressure[cmb_index]
    cmb_temperature = planet_mars.temperature[cmb_index]

    print(f"CMB Radius: {cmb_radius/1e3:.2f} km")
    print(f"CMB Pressure: {cmb_pressure/1e9:.2f} GPa")
    print(f"CMB Temperature: {cmb_temperature:.2f} K")
    """

    
    # Let's get MARSKB to compare everything to as we are trying to imitate Mars as seen from Insight

    marskb = burnman.seismic.MARSKB()
    marskbradii = 3389.5e3 - marskb.internal_depth_list() #s changed radius = 3389.5e3
   # marskbpressure = marskb.pressure(3389.5e3)

    # Kya change karna padega ?

    with warnings.catch_warnings(record=True) as w:
        eval = marskb.evaluate(["density", "pressure", "gravity", "v_s", "v_p"])
        marskbdensity, marskbpressure, marskbgravity, marskbvs, marskbvp = eval
        print(w[-1].message)




    # Now let's plot everything up

    # Optional prettier plotting

    plt.style.use('ggplot')

    figure = plt.figure(figsize=(10, 7))
    figure.suptitle(
        "{0} has a mass {1:.5f} times that of Earth,\n"
        "has an average density of {2:.3f} kg/m$^3$,\n"
        "and a moment of inertia factor of {3:.6f}".format(
            planet_mars.name,
            planet_mars.mass / 5.972e24,
            planet_mars.average_density,
            planet_mars.moment_of_inertia_factor,
        ),
        fontsize=12,
    )

    ax = [figure.add_subplot(2, 2, i) for i in range(1, 5)]

    bounds = np.array([[layer.radii[0]/1.e3, layer.radii[-1]/1.e3]
                   for layer in planet_mars.layers])

    # maxy = [8, 50, 6, 4000]  #s setting maximum values of Y axis

    layer_colors = ['#FF9A7F', '#FF99FF', '#99FF99', '#FFFFB3', '#8080C0']


    for bound, color in zip(bounds, layer_colors):
        for i in range(4):
            ax[i].axvspan(bound[0], bound[1], facecolor=color, alpha=0.25)


    ax[0].plot(planet_mars.radii / 1.0e3, planet_mars.density / 1.0e3, "k", linewidth=0.75, label=planet_mars.name)
    ax[0].plot(marskbradii / 1.0e3, marskbdensity / 1.0e3, "--k", linewidth=0.75, label="MARSKB")
    #ax[0].set_ylim(0.0, (max(planet_mars.density) / 1.0e3) + 1.0)
    ax[0].set_ylim(0.0, 10.0)
    ax[0].set_xlim(-1.0, 3400.0)
    ax[0].set_ylabel("Density ($\\cdot 10^3$ kg/m$^3$)")
    ax[0].legend()

    # Make a subplot showing the calculated pressure profile
    ax[1].plot(planet_mars.radii / 1.0e3, planet_mars.pressure / 1.0e9, "b", linewidth=0.75, label=planet_mars.name)
    #ax[1].plot(marskbradii / 1.0e3, marskbpressure, "--b", linewidth=0.75, label="MARSKB")
    #ax[1].set_ylim(0.0, (max(planet_mars.pressure) / 1e9) + 10.0)
    ax[1].set_xlim(-1.0, 3400.0)
    ax[1].set_ylim(0.0, 50.0)
    ax[1].set_ylabel("Pressure (GPa)")

    # Make a subplot showing the calculated gravity profile
    ax[2].plot(planet_mars.radii / 1.0e3, planet_mars.gravity, "g", linewidth=0.75)
    ax[2].plot(marskbradii / 1.0e3, marskbgravity, "--g", linewidth=0.75)
    #ax[2].set_ylim(0.0, max(planet_mars.gravity) + 0.5)
    ax[2].set_xlim(-1.0, 3400.0)
    ax[2].set_ylim(0.0, 6.0)
    ax[2].set_ylabel("Gravity (m/s$^2)$")
    

    # Make a subplot showing the calculated temperature profile
    mask = planet_mars.temperature > 100.0  #s
    ax[3].plot(planet_mars.radii[mask] / 1.0e3, planet_mars.temperature[mask], "r", linewidth=0.75)
    ax[3].set_ylabel("Temperature ($K$)")
    ax[3].set_xlabel("Radius (km)")
    #ax[3].set_ylim(0.0, max(planet_mars.temperature) + 100)
    ax[3].set_xlim(-1.0, 3400.0)
    ax[3].set_ylim(0.0, 4000)

    for i in range(2):
        ax[i].set_xticklabels([])
    for i in range(4):
        ax[i].set_xlim(0.0, max(planet_mars.radii) / 1.0e3)

    figure.set_tight_layout(True)
    plt.show()
