import math
import numpy as np
import multiprocessing

# from scipy.spatial import cKDTree : TODO -- faster?
from sklearn.neighbors import KDTree

import common as cm

#__add_to_archive(s, s.desc, archive, kdt)
def __add_to_archive(s, centroid, archive, kdt):
    niche_index = kdt.query([centroid], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = cm.make_hashable(niche) #change datatype
    s.centroid = n # coordinates of the centoid of the species s
    if n in archive: #if n is in the archive
        if s.fitness > archive[n].fitness: # if the fitness of s is higher thatn the fitness in the position of the s specie in the archieve
            archive[n] = s
            return 1
        return 0
    else: #if s is not in the archive, save it
        archive[n] = s #I add the new spcecie in the archive
        return 1


# evaluate a single vector (x) with a function f and return a species
# t = vector, function
def __evaluate(t):
    z, f = t  # evaluate z with function f, esto solo es para desempaquetar z y f
    fit, desc = f(z) # return fitness and descriptor 
    return cm.Species(z, desc, fit) #create a specie

# map-elites algorithm (CVT variant)
def compute(dim_map, dim_x, f,
            n_niches=1000,
            max_evals=1e5,
            params=cm.default_params,
            log_file=None,
            variation_operator=cm.variation):
    """CVT MAP-Elites
       Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3;22(4):623-30.

       Format of the logfile: evals archive_size max mean median 5%_percentile, 95%_percentile

    """
    # setup the parallel processing pool
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    # create the CVT
    c = cm.cvt(n_niches, dim_map, 
              params['cvt_samples'], params['cvt_use_cache']) #'cvt_use_cache': True or False, do we cache the cvt and reuse it
    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    cm.__write_centroids(c) #function in common que escribe en un file las coordenadas de los centroids en centroid_n_centroids_dimension_map

    archive = {} # init archive (empty)
    n_evals = 0 # number of evaluations since the beginning
    b_evals = 0 # number evaluation since the last dump

    # main loop
    while (n_evals < max_evals): 
        to_evaluate = []
        # random initialization
        if len(archive) <= params['random_init'] * n_niches: # Random_init: proportion of niches to be filled before starting
            for i in range(0, params['random_init_batch']): # batch for random initialization
                x = np.random.uniform(low=params['min'], high=params['max'], size=dim_x) # min/max value of parameters on the search space
                to_evaluate += [(x, f)]
        else:  # variation/selection loop
            keys = list(archive.keys()) #the coordinates of the archieve that are filled
            # we select all the parents at the same time because randint is slow
            rand1 = np.random.randint(len(keys), size=params['batch_size'])
            rand2 = np.random.randint(len(keys), size=params['batch_size'])
            for n in range(0, params['batch_size']):
                # parent selection
                x = archive[keys[rand1[n]]] #select various species
                y = archive[keys[rand2[n]]]
                # copy & add variation
                z = variation_operator(x.x, y.x, params) 
                to_evaluate += [(z, f)] #evaluate the children
        # evaluation of the fitness for to_evaluate
        s_list = cm.parallel_eval(__evaluate, to_evaluate, pool, params)
        # natural selection
        for s in s_list:  #each new species is check if it is added to the archive
            __add_to_archive(s, s.desc, archive, kdt)
        # count evals
        n_evals += len(to_evaluate)
        b_evals += len(to_evaluate)

        # write archive
        if b_evals >= params['dump_period'] and params['dump_period'] != -1:
            print("[{}/{}]".format(n_evals, int(max_evals)), end=" ", flush=True)
            cm.__save_archive(archive, n_evals)
            b_evals = 0
        # write log
        if log_file != None: #write in cvt file
            fit_list = np.array([x.fitness for x in archive.values()])
            log_file.write("{} {} {} {} {} {} {}\n".format(n_evals, len(archive.keys()),
                    fit_list.max(), np.mean(fit_list), np.median(fit_list),
                    np.percentile(fit_list, 5), np.percentile(fit_list, 95)))
            log_file.flush()
    cm.__save_archive(archive, n_evals)
    return archive
