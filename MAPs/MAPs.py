from cvt import compute
from Model_extended import model

default_params = \
    {
        # more of this -> higher-quality CVT
        "cvt_samples": 25000,
        # we evaluate in batches to paralleliez
        "batch_size": 100,
        # proportion of niches to be filled before starting
        "random_init": 0.1,
        # batch for random initialization
        "random_init_batch": 100,
        # when to write results (one generation = one batch)
        "dump_period": 10000,
        # do we use several cores?
        "parallel": True,
        # do we cache the result of CVT and reuse?
        "cvt_use_cache": True,
        # min/max of parameters
        "min": 0.1, ### ATENTION
        "max": 2, ### ATENTION
    }

archive = compute(3, 19, model, n_niches=800, max_evals=20000, log_file=open('cvt_20000.dat', 'w'), params = default_params)
#dim_map, dim_x, f, n_niches=1000,  max_evals=1e5, params=cm.default_params, log_file=None, variation_operator=cm.variation
