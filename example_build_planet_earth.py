# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2015 by the BurnMan team, released under the GNU
# GPL v2 or later.
"""
example_build_planet
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
import pandas as pd

import burnman

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

    # core
    core = burnman.Layer("core", radii=np.linspace(0, 1791.645e3, 10)) #s changed radius from Kh22
    core.set_material(burnman.minerals.other.Liquid_Fe_Anderson()) #s change material
    # The minerals that make up our core do not currently implement the thermal equation of state, so we will define the temperature at 300 K.
    core.set_temperature_mode("adiabatic", temperature_top=2100.0) #s 2100 frm S19 in Kh22 (#s core may not be adiabatic)

    # Next the Mantle

    mantle_radii = np.linspace(1791.645e3, 3346.303e3, 101) #s changed radius from Kh22
    mantle = burnman.Layer("mantle", radii=mantle_radii)
    mantle.set_material(burnman.minerals.SLB_2011.mg_bridgmanite()) #s change material (bridgmanite is less likely)

    # Here we add a thermal boundary layer perturbation, assuming that the
    # lower mantle has a Rayleigh number of 1.e7, and that there
    # is an 840 K jump across the basal thermal boundary layer and a
    # 0 K jump at the top of the lower mantle.

    tbl_perturbation = burnman.BoundaryLayerPerturbation(
        radius_bottom=1791.645e3, #s changed (this is radius of core)
        radius_top=3346.303e3, #s changed (this is radius of mantle)
        rayleigh_number=1.0e5, #s changed as Mars's stagnant lid mantle (current thermal state), also try 10^6 to see a mixed state of localized convection and stagnation.
        temperature_change=300.0, #s changed a/c paper! cite (300-600 K)
        boundary_layer_ratio=1.0, #s changed as sluggish mantle with top-heavy thermal structure  # 0.0 means No boundary layer i.e. uniform perturbation
    )

    mantle_tbl = tbl_perturbation.temperature(mantle_radii)
    mantle.set_temperature_mode(
        "perturbed-adiabatic", temperatures=mantle_tbl
    )

    crust = burnman.Layer(
        "crust", radii=np.linspace(3346.303e3, 3389.5e3, 10) #s changed radius from Kh22
    )
    crust.set_material(burnman.minerals.SLB_2011.forsterite())

    #s creating array for giving temp values from Drilleau et al, 2021 -> Dri21 (I'll use three set for aereotherm)
    crust_rads = np.array([3389.5, 3380.5, 3370.5, 3360.5, 3350.5, 3346.303])
    crust_temps = np.array([220, 243.1285, 268.8268, 294.5251, 320.2235, 331.0091])
    crust_temperatures = np.interp(crust.radii, crust_rads, crust_temps)
    # Set temperature mode to "user-defined" with the interpolated temperatures
    crust.set_temperature_mode("user-defined", crust_temperatures)
    
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
        "Planet Mars", [core, mantle, crust], verbose=True     #s omitted inner_core layer
    )
    print(planet_mars)

    # Here we compute its state. Go BurnMan Go!
    # (If we were to change composition of one of the layers, we would have to
    # recompute the state)
    planet_mars.make()

    # Now we output the mass of the planet and moment of inertia
    print(
        "\nmass/Earth= {0:.5f}, moment of inertia factor= {1:.6f}".format(
            planet_mars.mass/ 5.9722e24, planet_mars.moment_of_inertia_factor
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

    # Let's get MARSKB to compare everything to as we are trying to imitate Mars as seen from Insight
    marskb = burnman.seismic.MARSKB()
    marskbradii = 3389.5e3 - marskb.internal_depth_list() #s changed radius = 3389.5e3

    # Kya change karna padega ?
    with warnings.catch_warnings(record=True) as w:
        eval = marskb.evaluate(["density", "pressure", "gravity", "v_s", "v_p"])
        marskbdensity, marskbpressure, marskbgravity, marskbvs, marskbvp = eval
        print(w[-1].message)

    ###
    
    mars_radii = planet_mars.radii / 1.0e3
    mars_density = planet_mars.density / 1.0e3
    #mars_gravity = planet_mars.gravity           #s wrong
    
    df = pd.DataFrame({
    'Radius': mars_radii,
    'Density': mars_density
    #'Gravity': mars_gravity #s wrong
    })

    save_path = r"C:\Users\soumy\Downloads\planet_properties.csv"

    # Export to CSV
    
    df.to_csv(save_path, index=False)

    ###

    # Now let's plot everything up

    # Optional prettier plotting
    plt.style.use('ggplot')

    figure = plt.figure(figsize=(10, 12))
    figure.suptitle(
        "{0} has a mass {1:.4f} times that of Earth,\n"
        "has an average density of {2:.3f} kg/m$^3$,\n"
        "and a moment of inertia factor of {3:.6f}".format(
            planet_mars.name,
            planet_mars.mass / 5.9722e24,
            planet_mars.average_density,
            planet_mars.moment_of_inertia_factor,
        ),
        fontsize=15,
    )

    ax = [figure.add_subplot(4, 1, i) for i in range(1, 5)]

    ax[0].plot(
        planet_mars.radii / 1.0e3,
        planet_mars.density / 1.0e3,
        "k",
        linewidth=1.0,
        label=planet_mars.name,
    )
    ax[0].plot(
        marskbradii / 1.0e3, marskbdensity / 1.0e3, "--k", linewidth=1.0, label="MARSKB"
    )
    ax[0].set_ylim(0.0, (max(planet_mars.density) / 1.0e3) + 1.0)
    ax[0].set_ylabel("Density ($\\cdot 10^3$ kg/m$^3$)")
    ax[0].legend()

    # Make a subplot showing the calculated pressure profile
    ax[1].plot(
        planet_mars.radii / 1.0e3, 
        planet_mars.pressure / 1.0e9, 
        "b", 
        linewidth=1.0, 
        label=planet_mars.name,
    )
    ax[1].plot(marskbradii / 1.0e3, marskbpressure / 1.0e9, "--b", linewidth=1.0, label="MARSKB"
    )
    ax[1].set_ylim(0.0, (max(planet_mars.pressure) / 1e9) + 10.0)
    ax[1].set_ylabel("Pressure (GPa)")

    # Make a subplot showing the calculated gravity profile
    ax[2].plot(planet_mars.radii / 1.0e3, planet_mars.gravity, "g", linewidth=2.0)
    ax[2].plot(marskbradii / 1.0e3, marskbgravity, "--g", linewidth=1.0)
    ax[2].set_ylabel("Gravity (m/s$^2)$")
    ax[2].set_ylim(0.0, max(planet_mars.gravity) + 0.5)

    # Make a subplot showing the calculated temperature profile
    mask = planet_mars.temperature > 100.0  #s
    ax[3].plot(
        planet_mars.radii[mask] / 1.0e3, planet_mars.temperature[mask], "r", linewidth=1.0  #s
    )
    ax[3].set_ylabel("Temperature ($K$)")
    ax[3].set_xlabel("Radius (km)")
    ax[3].set_ylim(0.0, max(planet_mars.temperature) + 100)

    for i in range(3):
        ax[i].set_xticklabels([])
    for i in range(4):
        ax[i].set_xlim(0.0, max(planet_mars.radii) / 1.0e3)

    plt.show()
