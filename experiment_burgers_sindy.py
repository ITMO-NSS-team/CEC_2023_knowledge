import time
import numpy as np
import pandas as pd
import epde.interface.interface as epde_alg
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def draw_all_distributions(distrib1: dict, distrib2: dict, distrib3: dict):
    def make_names(names):
        ls = []
        for term in names:
            if len(term) == 2:
                ls.append(term[0] + ' * ' + term[1])
            else: ls.append(term[0])
        return  ls

    names = ['initial_fixed_dist', 'biased_dist', 'highly_biased_dist']
    count_members1 = list(distrib1.values())[:7]
    count_members2 = list(distrib2.values())[:7]
    count_members3 = list(distrib3.values())[:7]

    tags_ls = [1, 3, 5]
    terms_ls = list(distrib1.keys())[:7]
    categories = [(' ',) for i in range(len(terms_ls))]
    for i in range(len(terms_ls)):
        if i in tags_ls:
            categories[i] = terms_ls[i]
    categories = make_names(categories)#[i for i in range(len(distrib1))]

    x = np.array([1. + i * 0.04 for i in range(len(terms_ls))])
    width = 0.01

    fig, ax = plt.subplots(figsize=(14,8))
    ax.set_xticks(x)
    ax.set_xticklabels(categories)

    ax.bar(x-width/2, count_members1, width=width, alpha=0.5, label=names[0])
    ax.bar(x+width/2, count_members2, width=width, alpha=0.5, label=names[1])
    ax.bar(x+3*width/2, count_members3, width=width, alpha=0.5, label=names[2])

    plt.grid()
    plt.autoscale()
    plt.legend(prop={'size': 24}, framealpha=0.3)
    plt.xticks(fontsize=24, rotation=25)
    plt.yticks(fontsize=24)
    plt.ylabel('count', fontsize=24)
    # plt.savefig('data/distrib_burgers_sindy_7col.png')
    plt.show()


def draw_distribution(distrib: dict, title):
    distrib_ls = []
    idx = 0
    for value in distrib.values():
        ls_addition = [idx] * value
        distrib_ls += ls_addition
        idx += 1

    fig, ax1 = plt.subplots(figsize=(16, 8))
    sns.kdeplot(distrib_ls, ax=ax1)
    ax1.set_xlim(min(distrib_ls), max(distrib_ls))
    ax2 = ax1.twinx()
    sns.histplot(distrib_ls, ax=ax2, bins=len(distrib))  # discrete=True)
    plt.grid()
    # plt.savefig(f'data/{title}.png')
    plt.show()


def find_coeff_diff(res, coefficients: dict):
    differences = []

    for pareto_front in res:
        for soeq in pareto_front:
            if soeq.obj_fun[0] < 10:
                eq_text = soeq.vals.chromosome['u'].value.text_form
                terms_dict = out_formatting(eq_text)
                diff = coefficients_difference(terms_dict, coefficients)
                if diff != -1:
                    differences.append(diff)

    return differences


def coefficients_difference(terms_dict, coefficients):
    mae = 0.
    eq_found = 0
    for term_hash in terms_dict.keys():
        mae += abs(terms_dict.get(term_hash) - coefficients.get(term_hash))
        if coefficients.get(term_hash) != 0.0 and abs(terms_dict.get(term_hash) - coefficients.get(term_hash)) < 0.1:
            eq_found += 1

    mae /= len(terms_dict)
    if eq_found == 3:
        return mae
    else:
        return -1


def out_formatting(string):
    string = string.replace("u{power: 1.0}", "u")
    string = string.replace("d^2u/dx2^2{power: 1.0}", "d^2u/dx2^2")
    string = string.replace("d^2u/dx1^2{power: 1.0}", "d^2u/dx1^2")
    string = string.replace("du/dx1{power: 1.0}", "du/dx1")
    string = string.replace("du/dx2{power: 1.0}", "du/dx2")
    string = string.replace(" ", "")

    ls_equal = string.split('=')
    ls_left = ls_equal[0].split('+')
    ls_terms = []
    for term in ls_left:
        ls_term = term.split('*')
        ls_terms.append(ls_term)
    ls_right = ls_equal[1].split('*')

    terms_dict = {}
    for term in ls_terms:
        if len(term) == 1:
            terms_dict[1] = float(term[0])
        else:
            coeff = float(term.pop(0))
            terms_dict[hash_term(term)] = coeff

    terms_dict[hash_term(ls_right)] = -1.
    return terms_dict


def hash_term(term):
    total_term = 0
    for token in term:
        total_token = 1
        if type(token) == tuple:
            token = token[0]
        for char in token:
            total_token += ord(char)
        total_term += total_token * total_token
    return total_term


if __name__ == '__main__':

    distrib1 = {('u',): 60,
                ('du/dx1',): 84,
                ('du/dx2',): 73,
                ('d^2u/dx2^2',): 78,
                ('u', 'du/dx1'): 46,
                ('u', 'du/dx2'): 42,
                ('u', 'd^2u/dx2^2'): 45,
                ('du/dx1', 'du/dx2'): 48,
                ('du/dx1', 'd^2u/dx2^2'): 68,
                ('du/dx2', 'd^2u/dx2^2'): 52}
    distrib2 = {('u',): 60,
                ('du/dx1',): 100,
                ('du/dx2',): 73,
                ('d^2u/dx2^2',): 100,
                ('u', 'du/dx1'): 46,
                ('u', 'du/dx2'): 90,
                ('u', 'd^2u/dx2^2'): 45,
                ('du/dx1', 'du/dx2'): 48,
                ('du/dx1', 'd^2u/dx2^2'): 68,
                ('du/dx2', 'd^2u/dx2^2'): 52}
    distrib3 = {('u',): 60,
                ('du/dx1',): 230,
                ('du/dx2',): 73,
                ('d^2u/dx2^2',): 230,
                ('u', 'du/dx1'): 46,
                ('u', 'du/dx2'): 230,
                ('u', 'd^2u/dx2^2'): 45,
                ('du/dx1', 'du/dx2'): 48,
                ('du/dx1', 'd^2u/dx2^2'): 68,
                ('du/dx2', 'd^2u/dx2^2'): 52}
    distrib4 = {('u',): 1,
                ('du/dx1',): 1,
                ('du/dx2',): 1,
                ('d^2u/dx2^2',): 1,
                ('u', 'du/dx1'): 1,
                ('u', 'du/dx2'): 1,
                ('u', 'd^2u/dx2^2'): 1,
                ('du/dx1', 'du/dx2'): 1,
                ('du/dx1', 'd^2u/dx2^2'): 1,
                ('du/dx2', 'd^2u/dx2^2'): 1}

    # draw_all_distributions(distrib1, distrib2, distrib3)

    burg = loadmat('data/burgers.mat')
    t = np.ravel(burg['t'])
    x = np.ravel(burg['x'])
    u = np.real(burg['usol'])
    u = np.transpose(u)

    boundary = 10
    dimensionality = u.ndim
    grids = np.meshgrid(t, x, indexing='ij')

    ''' Parameters of the experiment '''
    write_csv = False
    print_results = True
    max_iter_number = 10
    distrib = {}  # {} or distrib1 or distrib2 or distrib3 or distrib4
    title = 'df0_sindy'


    terms = [('u',), ('du/dx1',), ('du/dx2',), ('d^2u/dx2^2',), ('u', 'du/dx1'), ('u', 'du/dx2'), ('u', 'd^2u/dx2^2'),
             ('du/dx1', 'du/dx2'), ('du/dx1', 'd^2u/dx2^2'), ('du/dx2', 'd^2u/dx2^2')]
    hashed_ls = [hash_term(term) for term in terms]
    coefficients = dict(zip(hashed_ls, [0., -1., 0., 0.1, 0., -1., 0., 0., 0., 0.]))
    coefficients[1] = 0.

    time_ls = []
    differences_ls = []
    num_found_eq = []
    for i in range(max_iter_number):
        epde_search_obj = epde_alg.epde_search(use_solver=False, boundary=boundary,
                                               dimensionality=dimensionality, coordinate_tensors=grids)

        epde_search_obj.set_moeadd_params(population_size=8, training_epochs=7)
        start = time.time()

        epde_search_obj.fit(data=u, max_deriv_order=(1, 2),
                            equation_terms_max_number=3, equation_factors_max_number=2,
                            coordinate_tensors=grids, eq_sparsity_interval=(1e-08, 1e-1),
                            memory_for_cache=25, prune_domain=True,
                            custom_prob_terms=distrib
                            )
        end = time.time()
        epde_search_obj.equation_search_results(only_print=True, level_num=4)
        time1 = end-start

        res = epde_search_obj.equation_search_results(only_print=False, level_num=4)
        difference_ls = find_coeff_diff(res, coefficients)

        if len(difference_ls) != 0:
            differences_ls.append(min(difference_ls))
        else:
            differences_ls.append(None)
        num_found_eq.append(len(difference_ls))
        print('Overall time is:', time1)
        print("Number of found eq:", len(difference_ls))
        print()
        time_ls.append(time1)

    if write_csv:
        arr = np.array([differences_ls, time_ls, num_found_eq])
        arr = arr.T
        df = pd.DataFrame(data=arr, columns=['MAE', 'time', 'number_found_eq'])
        df.to_csv(f'data/{title}.csv')
    if print_results:
        print('\nAverage time, s:', sum(time_ls) / len(time_ls))
        print('\nTime for every run:')
        for item in time_ls:
            print(item)
        print('\nMAE and # of found equations in every run:')
        for item1, item2 in zip(differences_ls, num_found_eq):
            print("diff:", item1, "num eq:", item2)

