import numpy as np
import pandas as pd
import burnman
from burnman import Composition, Mineral, Composite, Layer, Planet
import matplotlib.pyplot as plt

# Define your parameter bounds (example values, replace with your specific bounds)
parameter_bounds = {
    'inner_core_Fe': (90, 95),
    'outer_core_Fe': (88, 93),
    'lower_mantle_bridgmanite': (70, 90),
    'upper_mantle_olivine': (50, 70),
    'lithosphere_olivine': (85, 95),
    'crust_plagioclase': (30, 50)
}

# Number of iterations
num_iterations = 100

# Sampling function
def sample_parameters(bounds, num_samples):
    samples = {}
    for key, (low, high) in bounds.items():
        samples[key] = np.random.uniform(low, high, num_samples)
    return pd.DataFrame(samples)

# Generate parameter samples
samples = sample_parameters(parameter_bounds, num_iterations)

# Initialize results storage
results = []

for i in range(num_iterations):
    sample = samples.iloc[i]

    # Update compositions with sampled parameters
    inner_core_composition = Composition({'Fe': sample['inner_core_Fe'], 'Ni': 5., 'S': 0.10, 'Si': 0.30, 'O': 0.10}, 'weight')
    outer_core_composition = Composition({'Fe': sample['outer_core_Fe'], 'Ni': 5., 'S': 1.70, 'Si': 0.95, 'O': 0.80}, 'weight')

    inner_core_composition.renormalize('atomic', 'total', 1.)
    outer_core_composition.renormalize('atomic', 'total', 1.)

    # Creating materials using BurnMan
    inner_core_material = burnman.minerals.SE_2015.hcp_iron()
    outer_core_material = burnman.minerals.SE_2015.liquid_iron()

    # Inner and outer core layers
    inner_core = Layer('inner core', radii=np.linspace(0., 1220.e3, 21))
    outer_core = Layer('outer core', radii=np.linspace(1220.e3, 3480.e3, 21))

    inner_core.set_material(inner_core_material)
    outer_core.set_material(outer_core_material)

    inner_core.set_temperature_mode('adiabatic')
    outer_core.set_temperature_mode('adiabatic')

    # Mantle compositions and layers
    LM_bdg_fraction = sample['lower_mantle_bridgmanite'] / 100
    LM_rock = Composite([
        burnman.minerals.SLB_2024.bridgmanite(molar_fractions=[0.85, 0.10, 0.05, 0, 0, 0, 0]), 
        burnman.minerals.SLB_2024.ferropericlase(molar_fractions=[0.80, 0.20, 0, 0, 0]), 
        burnman.minerals.SLB_2024.capv()
    ], [LM_bdg_fraction, 0.09, 1 - LM_bdg_fraction - 0.09])

    lower_mantle = Layer('lower mantle', radii=np.linspace(3480.e3, 5701.e3, 50))
    lower_mantle.set_material(LM_rock)
    lower_mantle.set_temperature_mode('adiabatic')

    """
    UM_ol_fraction = sample['upper_mantle_olivine'] / 100
    UM_rock = Composite([
        burnman.minerals.SLB_2024.olivine(molar_fractions=[0.90, 0.10]),
        burnman.minerals.SLB_2024.orthopyroxene(molar_fractions = [0.85, 0.15, 0, 0]),
        burnman.minerals.SLB_2024.clinopyroxene(molar_fractions = [0.80, 0.00, 0.05, 0.15, 0, 0]),
        burnman.minerals.SLB_2024.mg_fe_aluminous_spinel(molar_fractions = [0.80, 0.10, 0.05, 0.05]),
        burnman.minerals.SLB_2024.garnet(molar_fractions = [0.85, 0.10, 0.05, 0, 0, 0, 0])
    ], [UM_ol_fraction, 0.20, 0.10, 0.05, 0.65 - UM_ol_fraction])
    """

    # Ensure the total is exactly 1.0, adjust others proportionally if necessary:
    UM_ol_fraction = sample['upper_mantle_olivine'] / 100

    # Fixed fractions for other minerals (example: orthopyroxene, clinopyroxene, spinel)
    opx_fraction = 0.20
    cpx_fraction = 0.10
    spinel_fraction = 0.05

    # Calculate remaining fraction for garnet, making sure it is not negative
    remaining_fraction = 1.0 - (UM_ol_fraction + opx_fraction + cpx_fraction + spinel_fraction)

    if remaining_fraction < 0:
        # Adjust the UM_ol_fraction to ensure it doesn't exceed the valid range
        UM_ol_fraction = 1.0 - (opx_fraction + cpx_fraction + spinel_fraction)
        remaining_fraction = 0.0

    # Now assign fractions to UM_rock
    UM_rock = Composite([
        burnman.minerals.SLB_2024.olivine(molar_fractions=[0.90, 0.10]),
        burnman.minerals.SLB_2024.orthopyroxene(molar_fractions = [0.85, 0.15, 0, 0]),
        burnman.minerals.SLB_2024.clinopyroxene(molar_fractions = [0.80, 0.00, 0.05, 0.15, 0, 0]),
        burnman.minerals.SLB_2024.mg_fe_aluminous_spinel(molar_fractions = [0.80, 0.10, 0.05, 0.05]),
        burnman.minerals.SLB_2024.garnet(molar_fractions = [0.85, 0.10, 0.05, 0, 0, 0, 0])
    ], [UM_ol_fraction, opx_fraction, cpx_fraction, spinel_fraction, remaining_fraction])


    upper_mantle = Layer('upper mantle', radii=np.linspace(5701.e3, 6151.e3, 50))
    upper_mantle.set_material(UM_rock)
    upper_mantle.set_temperature_mode('adiabatic')

    # Lithosphere composition
    lith_ol_fraction = sample['lithosphere_olivine'] / 100
    dunite = burnman.minerals.SLB_2011.mg_fe_olivine(molar_fractions=[lith_ol_fraction, 1 - lith_ol_fraction])

    lithosphere = Layer('lithosphere', radii=np.linspace(6151.e3, 6346.e3, 50))
    lithosphere.set_material(dunite)
    lithosphere.set_temperature_mode('user-defined', np.linspace(1548., 614., 50))

    # Crust composition
    crust_plag_fraction = sample['crust_plagioclase'] / 100
    crust_material = burnman.minerals.SLB_2011.plagioclase(molar_fractions=[crust_plag_fraction, 1 - crust_plag_fraction])

    crust = Layer('crust', radii=np.linspace(6346.e3, 6371.e3, 25))
    crust.set_material(crust_material)
    crust.set_temperature_mode('user-defined', np.linspace(614., 300., 25))

    # Planet construction
    model_earth = Planet('Sampled Earth', [inner_core, outer_core, lower_mantle, upper_mantle, lithosphere, crust])
    model_earth.make()

    # Storing results
    results.append({
        **sample,
        'mass': model_earth.mass,
        'moment_of_inertia_factor': model_earth.moment_of_inertia_factor
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv("earth_model_parameter_exploration.csv", index=False)

# Plotting results
fig, axes = plt.subplots(len(parameter_bounds), 1, figsize=(12, 18), constrained_layout=True)
fig.suptitle("Parameter Exploration Results with Bounds", fontsize=16)

for idx, param in enumerate(parameter_bounds.keys()):
    axes[idx].hist(results_df[param], bins=25, alpha=0.7, color='skyblue', edgecolor='black')
    axes[idx].axvline(parameter_bounds[param][0], color='red', linestyle='--', label='Lower Bound')
    axes[idx].axvline(parameter_bounds[param][1], color='green', linestyle='--', label='Upper Bound')
    axes[idx].set_xlabel(param, fontsize=12)
    axes[idx].set_ylabel("Frequency", fontsize=12)
    axes[idx].legend()

plt.show()
