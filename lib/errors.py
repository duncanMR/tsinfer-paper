import numpy as np
import sgkit

def fetch_empirical_probs(freq, df):
    error_freq = df.freq.values
    row = df.loc[np.searchsorted(error_freq, freq)]
    probs = np.array([
        [row.p00,  0.5 * row.p01, 0.5 * row.p01, row.p02],
        [row.p10,  row.p11,       0,             row.p12],
        [row.p10,  0,             row.p11,       row.p12],
        [row.p20,  0.5 * row.p21, 0.5 * row.p21, row.p22],
    ])
    assert np.all(np.isclose(probs.sum(axis=1), 1.0, atol=1e-5))
    return probs

def encode_genotypes(genotypes):
    return genotypes[:, 0] * 2 + genotypes[:, 1]

def decode_genotypes(idx):
    genotypes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
    return genotypes[idx]

def resample_genotype(genotype, probs, rng):
    assert len(genotype) == 2
    input_idx = encode_genotypes(np.array([genotype]))[0]
    output_idx = rng.choice(len(probs), p=probs[input_idx])
    return decode_genotypes(output_idx)

def resample_genotypes_vectorized(genotypes, probs, rng):
    input_idx = encode_genotypes(genotypes)
    cum_probs = np.cumsum(probs, axis=1)
    U = rng.random(len(input_idx))
    output_idx = (U[:, None] < cum_probs[input_idx]).argmax(axis=1)
    return decode_genotypes(output_idx)

def add_genotype_errors(call_genotype, rng, probs_func, **kwargs):
    #no multiallelic sites
    assert len(np.unique(call_genotype)) <= 2 
    assert len(call_genotype.shape) == 3
    assert call_genotype.shape[2] == 2

    G_out = np.full_like(call_genotype, 0, dtype=np.int8)
    num_sites = G_out.shape[0]
    num_samples = G_out.shape[1]
    an = num_samples*2 

    for site in range(num_sites):
        g = call_genotype[site, :, :]
        freq = np.sum(g)/an
        probs = probs_func(freq, **kwargs)
        G_out[site, :, :] = resample_genotypes_vectorized(g, probs, rng)

    return G_out

def temp_add_genotype_errors(ds, output_path):
    sgkit.save_dataset(ds, output_path.parent)
    output_path.touch()