import stdpopsim 


def simulate(model, contig, samples, left, right, seed):
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model(model)
    contig = species.get_contig(
        contig,
        mutation_rate=model.mutation_rate,
        left=float(left),
        right=float(right),
        genetic_map="HapMapII_GRCh38",
    )
    engine = stdpopsim.get_engine("msprime")
    arg = engine.simulate(model, contig, samples, seed=seed, msprime_model="smc_prime")
    return arg