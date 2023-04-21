"""
Library with functions that checks simple assumptions of genomic data:

- Frequencia alêlica
- Fixation Index fst
- Harvey-Weinberg equilibrium
"""


import numpy as np


def fst(genome):
    """
    Calculates the generated genome Fixation Index

    Args:
        genome (Dataframe): Genotype data that we desire to check
    """

    pass


def harvey_weinberg(genome):
    """
    Checks if the haplotypes of the genomes follow the Harvey-Weinberg equilibrium

    Args:
        genome (Dataframe): Genotype data that we desire to check
    """

    A = np.unique(genome, return_counts=True)
