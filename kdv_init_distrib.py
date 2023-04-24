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
    count_members1 = list(distrib1.values())[:9]
    count_members2 = list(distrib2.values())[:9]
    count_members3 = list(distrib3.values())[:9]

    tags_ls = [1, 4, 5, 7]
    terms_ls = list(distrib1.keys())[:9]
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
    # plt.savefig('data_kdv/distrib_kdv_9col.png')
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
    sns.kdeplot(distrib_ls, ax=ax1)
    ax1.set_xlim(min(distrib_ls), max(distrib_ls))
    ax2 = ax1.twinx()
    sns.histplot(distrib_ls, ax=ax2, bins=len(distrib))  # discrete=True)
    plt.grid()
    if save_plot:
        plt.savefig(f'data_kdv/{title}.png')
    plt.show()


coefficients_direct2 = {('u',): 0.,
                ('du/dx1',): -1.,
                ('du/dx2',): 0.,
                ('d^2u/dx2^2',): 0.,
                ('d^3u/dx2^3',): -1.,
                ('cos(t)sin(x)',): 1.,
                ('u', 'du/dx1'): 0.,
                ('u', 'du/dx2'): -6.,
                ('u', 'd^2u/dx2^2'): 0.,
                ('u', 'd^3u/dx2^3'): 0.,
                ('u', 'cos(t)sin(x)'): 0.,
                ('du/dx1', 'du/dx2'): 0.,
                ('du/dx1', 'd^2u/dx2^2'): 0.,
                ('du/dx1', 'd^3u/dx2^3'): 0.,
                ('du/dx1', 'cos(t)sin(x)'): 0.,
                ('du/dx2', 'd^2u/dx2^2'): 0.,
                ('du/dx2', 'd^3u/dx2^3'): 0.,
                ('du/dx2', 'cos(t)sin(x)'): 0.,
                ('d^2u/dx2^2', 'd^3u/dx2^3'): 0.,
                ('d^2u/dx2^2', 'cos(t)sin(x)'): 0.,
                ('d^3u/dx2^3', 'cos(t)sin(x)'): 0.,
                }

coefficients_direct1 = {('u',): 0.,
                ('du/dx1',): -0.16666666666667,
                ('du/dx2',): 0.,
                ('d^2u/dx2^2',): 0.,
                ('d^3u/dx2^3',): -0.16666666666667,
                ('cos(t)sin(x)',): 0.16666666666667,
                ('u', 'du/dx1'): 0.,
                ('u', 'du/dx2'): -1.,
                ('u', 'd^2u/dx2^2'): 0.,
                ('u', 'd^3u/dx2^3'): 0.,
                ('u', 'cos(t)sin(x)'): 0.,
                ('du/dx1', 'du/dx2'): 0.,
                ('du/dx1', 'd^2u/dx2^2'): 0.,
                ('du/dx1', 'd^3u/dx2^3'): 0.,
                ('du/dx1', 'cos(t)sin(x)'): 0.,
                ('du/dx2', 'd^2u/dx2^2'): 0.,
                ('du/dx2', 'd^3u/dx2^3'): 0.,
                ('du/dx2', 'cos(t)sin(x)'): 0.,
                ('d^2u/dx2^2', 'd^3u/dx2^3'): 0.,
                ('d^2u/dx2^2', 'cos(t)sin(x)'): 0.,
                ('d^3u/dx2^3', 'cos(t)sin(x)'): 0.,
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
             ('cos(t)sin(x)',): 1,
             ('u', 'du/dx1'): 1,
             ('u', 'du/dx2'): 1,
             ('u', 'd^2u/dx2^2'): 1,
             ('u', 'd^3u/dx2^3'): 1,
             ('u', 'cos(t)sin(x)'): 1,
             ('du/dx1', 'du/dx2'): 1,
             ('du/dx1', 'd^2u/dx2^2'): 1,
             ('du/dx1', 'd^3u/dx2^3'): 1,
             ('du/dx1', 'cos(t)sin(x)'): 1,
             ('du/dx2', 'd^2u/dx2^2'): 1,
             ('du/dx2', 'd^3u/dx2^3'): 1,
             ('du/dx2', 'cos(t)sin(x)'): 1,
             ('d^2u/dx2^2', 'd^3u/dx2^3'): 1,
             ('d^2u/dx2^2', 'cos(t)sin(x)'): 1,
             ('d^3u/dx2^3', 'cos(t)sin(x)'): 1}

distrib1 = {('u',) : 78,
            ('du/dx1',) : 104,
            ('du/dx2',) : 99,
            ('d^2u/dx2^2',) : 105,
            ('d^3u/dx2^3',) : 108,
            ('cos(t)sin(x)',) : 62,
            ('u', 'du/dx1') : 33,
            ('u', 'du/dx2') : 37,
            ('u', 'd^2u/dx2^2') : 31,
            ('u', 'd^3u/dx2^3') : 32,
            ('u', 'cos(t)sin(x)') : 24,
            ('du/dx1', 'du/dx2') : 34,
            ('du/dx1', 'd^2u/dx2^2') : 41,
            ('du/dx1', 'd^3u/dx2^3') : 34,
            ('du/dx1', 'cos(t)sin(x)') : 34,
            ('du/dx2', 'd^2u/dx2^2') : 55,
            ('du/dx2', 'd^3u/dx2^3') : 48,
            ('du/dx2', 'cos(t)sin(x)') : 36,
            ('d^2u/dx2^2', 'd^3u/dx2^3') : 49,
            ('d^2u/dx2^2', 'cos(t)sin(x)') : 31,
            ('d^3u/dx2^3', 'cos(t)sin(x)') : 25}


distrib2 = {('u',) : 78,
            ('du/dx1',) : 106,     ###
            ('du/dx2',) : 99,
            ('d^2u/dx2^2',) : 105,
            ('d^3u/dx2^3',) : 108,    ###
            ('cos(t)sin(x)',) : 97,   ###
            ('u', 'du/dx1') : 33,
            ('u', 'du/dx2') : 90,     ###
            ('u', 'd^2u/dx2^2') : 31,
            ('u', 'd^3u/dx2^3') : 32,
            ('u', 'cos(t)sin(x)') : 24,
            ('du/dx1', 'du/dx2') : 34,
            ('du/dx1', 'd^2u/dx2^2') : 41,
            ('du/dx1', 'd^3u/dx2^3') : 34,
            ('du/dx1', 'cos(t)sin(x)') : 34,
            ('du/dx2', 'd^2u/dx2^2') : 55,
            ('du/dx2', 'd^3u/dx2^3') : 48,
            ('du/dx2', 'cos(t)sin(x)') : 36,
            ('d^2u/dx2^2', 'd^3u/dx2^3') : 49,
            ('d^2u/dx2^2', 'cos(t)sin(x)') : 31,
            ('d^3u/dx2^3', 'cos(t)sin(x)') : 25}


distrib3 = {('u',) : 78,
            ('du/dx1',) : 250,     ###
            ('du/dx2',) : 99,
            ('d^2u/dx2^2',) : 105,
            ('d^3u/dx2^3',) : 250,    ###
            ('cos(t)sin(x)',) : 250,   ###
            ('u', 'du/dx1') : 33,
            ('u', 'du/dx2') : 250,     ###
            ('u', 'd^2u/dx2^2') : 31,
            ('u', 'd^3u/dx2^3') : 32,
            ('u', 'cos(t)sin(x)') : 24,
            ('du/dx1', 'du/dx2') : 34,
            ('du/dx1', 'd^2u/dx2^2') : 41,
            ('du/dx1', 'd^3u/dx2^3') : 34,
            ('du/dx1', 'cos(t)sin(x)') : 34,
            ('du/dx2', 'd^2u/dx2^2') : 55,
            ('du/dx2', 'd^3u/dx2^3') : 48,
            ('du/dx2', 'cos(t)sin(x)') : 36,
            ('d^2u/dx2^2', 'd^3u/dx2^3') : 49,
            ('d^2u/dx2^2', 'cos(t)sin(x)') : 31,
            ('d^3u/dx2^3', 'cos(t)sin(x)') : 25}

# draw_distribution(distrib3, "title")
draw_all_distributions(distrib1, distrib2, distrib3)
