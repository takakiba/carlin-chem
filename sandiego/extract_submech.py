def extract_c1():
    import cantera as ct

    input_file = 'chem.yaml'
    all_species = ct.Species.listFromFile(input_file)
    species = []

    # Filter species
    for S in all_species:
        comp = S.composition
        if 'C' in comp and comp['C'] >1:
            # Exclude all hydrocarbon species
            continue
        if 'N' in comp and comp != {'N': 2}:
            # Exclude all nitrogen compounds except for N2
            continue
        if 'Ar' in comp:
            # Exclude Argon
            continue
        species.append(S)

    species_names = {S.name for S in species}

    # Filter reactions, keeping only those that only involve the selected species
    ref_phase = ct.Solution(thermo='ideal-gas', kinetics='gas', species=all_species)
    all_reactions = ct.Reaction.listFromFile(input_file, ref_phase)
    reactions = []

    for R in all_reactions:
        if not all(reactant in species_names for reactant in R.reactants):
            continue

        if not all(product in species_names for product in R.products):
            continue

        reactions.append(R)

    gas = ct.Solution(thermo='ideal-gas', kinetics='gas',
                       species=species, reactions=reactions)
    return gas

