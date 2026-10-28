"""Copy and edit this workflow to customize indexing.

Run from your data directory: python /path/to/custom_find_orientations.py config.yml
Stage methods handle configuration, caching, parallelism, and progress reporting.
"""
import argparse
import os

from hexrd.hedm.experiment import HedmExperiment
from hexrd.hedm.find_orientations import (
    FindOrientationsResult,
    cluster_grains,
    generate_trial_orientations,
    load_or_build_eta_omega_maps,
    load_plane_data,
    score_orientations,
    select_min_samples,
    write_results,
)


def find_orientations(experiment, material=None, clean=False):
    """Replace or modify any stage here to build your own workflow."""
    plane_data = load_plane_data(experiment, material)
    maps = load_or_build_eta_omega_maps(experiment, plane_data, clean=clean)
    orientations = generate_trial_orientations(experiment, plane_data, maps)
    min_samples = select_min_samples(experiment, plane_data, maps, orientations)
    completeness = score_orientations(experiment, plane_data, maps, orientations)
    grains = cluster_grains(experiment, plane_data, orientations, completeness,
                            min_samples=min_samples)
    return FindOrientationsResult(grains, orientations, completeness)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='HEDM YAML configuration')
    parser.add_argument('--clean', action='store_true', help='rebuild eta-omega maps')
    args = parser.parse_args()
    experiment = HedmExperiment(args.config, working_dir=os.getcwd())
    result = find_orientations(experiment, clean=args.clean)
    output_dir = write_results(result, experiment)
    print(f'Found {result.num_grains} grain(s); results written to {output_dir}')
