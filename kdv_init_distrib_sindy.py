import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def draw_all_distributions(distrib1: dict, distrib2: dict, distrib3: dict):

    def make_names(names):
        ls = []
        for term in names:
            if len(term) == 2:
                ls.append(term[0] + ' * ' + term[1])
            else: ls.append(term[0])
        return ls

    names = ['initial_fixed_dist', 'biased_dist', 'highly_biased_dist']
    count_members1 = list(distrib1.values())[:8]
    count_members2 = list(distrib2.values())[:8]
    count_members3 = list(distrib3.values())[:8]

    tags_ls = [1, 4, 6]
    terms_ls = list(distrib1.keys())[:8]
    categories = [(' ',) for i in range(len(terms_ls))]
    for i in range(len(terms_ls)):
        if i in tags_ls:
            categories[i] = terms_ls[i]

    categories = make_names(categories)

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
    # plt.savefig('data_kdv/distrib_kdv_sindy_8col.png')
    plt.show()


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


def draw_distribution(distrib: dict, title, save_plot=False):
    distrib_ls = []
    idx = 0
    for value in distrib.values():
        ls_addition = [idx] * value
        distrib_ls += ls_addition
        idx += 1

    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.set_xlim(min(distrib_ls), max(distrib_ls))
    sns.histplot(distrib_ls, ax=ax1, bins=len(distrib))  # discrete=True)
    plt.grid()
    if save_plot:
        plt.savefig(f'data_kdv/{title}.png')
    plt.show()


coefficients_direct2 = {('u',): 0.,
                ('du/dx1',): -1.,
                ('du/dx2',): 0.,
                ('d^2u/dx2^2',): 0.,
                ('d^3u/dx2^3',): -1.,
                ('u', 'du/dx1'): 0.,
                ('u', 'du/dx2'): -6.,
                ('u', 'd^2u/dx2^2'): 0.,
                ('u', 'd^3u/dx2^3'): 0.,
                ('du/dx1', 'du/dx2'): 0.,
                ('du/dx1', 'd^2u/dx2^2'): 0.,
                ('du/dx1', 'd^3u/dx2^3'): 0.,
                ('du/dx2', 'd^2u/dx2^2'): 0.,
                ('du/dx2', 'd^3u/dx2^3'): 0.,
                ('d^2u/dx2^2', 'd^3u/dx2^3'): 0.,
                }

coefficients_direct1 = {('u',): 0.,
                ('du/dx1',): -0.16666666666667,
                ('du/dx2',): 0.,
                ('d^2u/dx2^2',): 0.,
                ('d^3u/dx2^3',): -0.16666666666667,
                ('u', 'du/dx1'): 0.,
                ('u', 'du/dx2'): -1.,
                ('u', 'd^2u/dx2^2'): 0.,
                ('u', 'd^3u/dx2^3'): 0.,
                ('du/dx1', 'du/dx2'): 0.,
                ('du/dx1', 'd^2u/dx2^2'): 0.,
                ('du/dx1', 'd^3u/dx2^3'): 0.,
                ('du/dx2', 'd^2u/dx2^2'): 0.,
                ('du/dx2', 'd^3u/dx2^3'): 0.,
                ('d^2u/dx2^2', 'd^3u/dx2^3'): 0.,
                }
term_ls = list(coefficients_direct1.keys())
values = list(coefficients_direct1.values())
hashed_ls = [hash_term(term) for term in term_ls]

values2 = list(coefficients_direct2.values())

coefficients1 = dict(zip(hashed_ls, values))
coefficients1[1] = 0.
coefficients2 = dict(zip(hashed_ls, values2))
coefficients2[1] = 0.

distrib4 = { ('u',): 1,
             ('du/dx1',): 1,
             ('du/dx2',): 1,
             ('d^2u/dx2^2',): 1,
             ('d^3u/dx2^3',): 1,
             ('u', 'du/dx1'): 1,
             ('u', 'du/dx2'): 1,
             ('u', 'd^2u/dx2^2'): 1,
             ('u', 'd^3u/dx2^3'): 1,
             ('du/dx1', 'du/dx2'): 1,
             ('du/dx1', 'd^2u/dx2^2'): 1,
             ('du/dx1', 'd^3u/dx2^3'): 1,
             ('du/dx2', 'd^2u/dx2^2'): 1,
             ('du/dx2', 'd^3u/dx2^3'): 1,
             ('d^2u/dx2^2', 'd^3u/dx2^3'): 1}

distrib1 = {('u',): 633,
            ('du/dx1',): 994,
            ('du/dx2',): 1000,
            ('d^2u/dx2^2',): 973,
            ('d^3u/dx2^3',): 941,
            ('u', 'du/dx1'): 407,
            ('u', 'du/dx2'): 453,
            ('u', 'd^2u/dx2^2'): 442,
            ('u', 'd^3u/dx2^3'): 424,
            ('du/dx1', 'du/dx2'): 519,
            ('du/dx1', 'd^2u/dx2^2'): 491,
            ('du/dx1', 'd^3u/dx2^3'): 464,
            ('du/dx2', 'd^2u/dx2^2'): 466,
            ('du/dx2', 'd^3u/dx2^3'): 461,
            ('d^2u/dx2^2', 'd^3u/dx2^3'): 469}


distrib2 = {('u',): 633,
            ('du/dx1',): 1300,
            ('du/dx2',): 1000,
            ('d^2u/dx2^2',): 973,
            ('d^3u/dx2^3',): 1280,
            ('u', 'du/dx1'): 407,
            ('u', 'du/dx2'): 1000,
            ('u', 'd^2u/dx2^2'): 442,
            ('u', 'd^3u/dx2^3'): 424,
            ('du/dx1', 'du/dx2'): 519,
            ('du/dx1', 'd^2u/dx2^2'): 491,
            ('du/dx1', 'd^3u/dx2^3'): 464,
            ('du/dx2', 'd^2u/dx2^2'): 466,
            ('du/dx2', 'd^3u/dx2^3'): 461,
            ('d^2u/dx2^2', 'd^3u/dx2^3'): 469}


distrib3 = {('u',): 633,
            ('du/dx1',): 2500,
            ('du/dx2',): 1000,
            ('d^2u/dx2^2',): 973,
            ('d^3u/dx2^3',): 2500,
            ('u', 'du/dx1'): 407,
            ('u', 'du/dx2'): 2200,
            ('u', 'd^2u/dx2^2'): 442,
            ('u', 'd^3u/dx2^3'): 424,
            ('du/dx1', 'du/dx2'): 519,
            ('du/dx1', 'd^2u/dx2^2'): 491,
            ('du/dx1', 'd^3u/dx2^3'): 464,
            ('du/dx2', 'd^2u/dx2^2'): 466,
            ('du/dx2', 'd^3u/dx2^3'): 461,
            ('d^2u/dx2^2', 'd^3u/dx2^3'): 469}

# draw_distribution(distrib3, "title1")
draw_all_distributions(distrib1, distrib2, distrib3)
