from math import gamma, pi, sin
import numpy as np
from random import normalvariate, randint, random
from pathlib import Path


class sw(object):

    def __init__(self):

        self.__Positions = []
        self.__Gbest = []
        self.__best_fitness = None

    def _set_Gbest(self, Gbest):
        self.__Gbest = Gbest

    def _points(self, agents):
        self.__Positions.append([list(i) for i in agents])

    def get_agents(self):
        """Returns a history of all agents of the algorithm (return type:
        list)"""

        return self.__Positions

    def get_Gbest(self):
        """Return the best position of algorithm (return type: list)"""
        return list(self.__Gbest)

    def _set_best_fitness(self, fitness):
        self.__best_fitness = fitness

    def get_best_fitness(self):
        """Return the best fitness of algorithm (return type: foat)"""
        return self.__best_fitness


class cso(sw):
    """
    Cuckoo Search Optimization
    """

    def __init__(self, wavelet, n, function, lb, ub, dimension, iteration, pa=0.25,
                 nest=100, discrete=None, minimize=True):
        """
        :param n: number of agents
        :param function: test function
        :param lb: int or array-like, lower limits for plot axes
        :param ub: int or array-like, upper limits for plot axes
        :param dimension: space dimension
        :param iteration: number of iterations
        :param pa: probability of cuckoo's egg detection (default value is 0.25)
        :param nest: number of nests (default value is 100)
        :param discrete: None or or array-like (default value is None), indicate that some parameters are discrete
        """
        super(cso, self).__init__()

        if discrete:
            self.discrete = np.array(discrete)
        else:
            self.discrete = np.zeros(dimension, dtype=bool)

        self.__Nests = []
        self.__agents = np.random.uniform(lb, ub, (n, dimension))
        self.__nests = np.random.uniform(lb, ub, (nest, dimension))

        self.__agents[:, self.discrete] = np.round(self.__agents[:, self.discrete])
        self.__nests[:, self.discrete] = np.round(self.__nests[:, self.discrete])

        _nests_fitness = np.array([function(x) for x in self.__nests])
        Pbestidx = _nests_fitness.argmin()
        Pbest = self.__nests[Pbestidx]
        Gbest = Pbest
        Gbest_fittness = _nests_fitness[Pbestidx]

        '''
        save log
        '''
        if minimize:
            save_log = Path('./log/price/{}'.format(wavelet))
        else:
            save_log = Path('./log/binary/{}'.format(wavelet))
        save_log.mkdir(exist_ok=True)
        save_log = save_log / 'Pbest_log.txt'
        with open(save_log, 'a+') as f:
            f.write('fitness={}\ndenoise={}, lstm_units={}, lstm_act_f={},  lstm_layer={}, lstm_dropout={}, lstm_recurrent_dropout={}, att={}, dense_unit={}, dense_layer={}, dense_act_f={}, dense_drop={}, batch_size={}\n\n'.format(Gbest_fittness, *Gbest))

        self._points(self.__agents)

        for _ in range(iteration):

            _agents_fitness = np.array([function(x) for x in self.__agents])

            for i in range(n):
                val = randint(0, nest - 1)
                if minimize:
                    if _agents_fitness[i] < _nests_fitness[val]:
                        self.__nests[val] = self.__agents[i]
                        _nests_fitness[val] = _agents_fitness[i]
                else:
                    if _agents_fitness[i] > _nests_fitness[val]:
                        self.__nests[val] = self.__agents[i]
                        _nests_fitness[val] = _agents_fitness[i]

            # 一部分糟糕的巢被拋棄
            discovered = self.__empty_nests(pa, lb, ub, nest, dimension, Pbestidx)
            _new_fitness = list()
            for d in range(nest):
                if discovered[d]:
                    _new_fitness.append(function(self.__nests[d]))
                else:
                    _new_fitness.append(_nests_fitness[d])
            _nests_fitness = np.array(_new_fitness)

            fcuckoos = [(_agents_fitness[i], i) for i in range(n)]
            fnests = [(_nests_fitness[i], i) for i in range(nest)]
            if minimize:
                fcuckoos.sort(reverse=True)
                fnests.sort()
            else:
                fcuckoos.sort()
                fnests.sort(reverse=True)

            if nest > n:
                mworst = n
            else:
                mworst = nest

            for i in range(mworst):
                if minimize:
                    if fnests[i][0] < fcuckoos[i][0]:
                        self.__agents[fcuckoos[i][1]] = self.__nests[fnests[i][1]]
                else:
                    if fnests[i][0] > fcuckoos[i][0]:
                        self.__agents[fcuckoos[i][1]] = self.__nests[fnests[i][1]]

            self.__Levyfly(Pbest, lb, ub, n, dimension)
            self._points(self.__agents)
            self.__nest()
            Pbestidx = _nests_fitness.argmin()
            Pbest = self.__nests[Pbestidx]
            if minimize:
                if _nests_fitness[Pbestidx] < Gbest_fittness:
                    Gbest = Pbest
                    Gbest_fittness = _nests_fitness[Pbestidx]
            else:
                if _nests_fitness[Pbestidx] > Gbest_fittness:
                    Gbest = Pbest
                    Gbest_fittness = _nests_fitness[Pbestidx]

                with open(save_log, 'a+') as f:
                    f.write('fitness={}\ndenoise={}, lstm_units={}, lstm_act_f={},  lstm_layer={}, lstm_dropout={}, lstm_recurrent_dropout={}, att={}, dense_unit={}, dense_layer={}, dense_act_f={}, dense_drop={}, batch_size={}\n\n'.format(Gbest_fittness, *Gbest))

        self._set_Gbest(Gbest)
        self._set_best_fitness(Gbest_fittness)

    def __nest(self):
        self.__Nests.append([list(i) for i in self.__nests])

    def __drawLevy(self, dimension):
        beta = 1.5
        sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=dimension)
        v = np.random.normal(0, 1, size=dimension)
        return u / abs(v) ** (1 / beta)

    def __Levyfly(self, Pbest, lb, ub, n, dimension):
        for i in range(n):
            step = self.__drawLevy(dimension)
            stepsize = 0.02 * step[~self.discrete] * (self.__agents[i][~self.discrete] - Pbest[~self.discrete])
            discrete_stepsize = 0.4 * step[self.discrete] * (self.__agents[i][self.discrete] - Pbest[self.discrete])
            self.__agents[i][~self.discrete] += stepsize * np.array([normalvariate(0, 1) for k in range(sum(~self.discrete))])
            self.__agents[i][self.discrete] += discrete_stepsize * np.array([normalvariate(0, 1) for k in range(sum(self.discrete))])

        self.__agents = np.clip(self.__agents, lb, ub)
        self.__agents[:, self.discrete] = np.round(self.__agents[:, self.discrete])

    def __empty_nests(self, pa, lb, ub,  n, dimension, Pbestidx):
        # A fraction of worse nests are discovered with a probability pa
        # Discovered or not -- a status vector
        discovered = np.random.rand(n) < pa
        discovered[Pbestidx] = False
        stepsize = np.random.rand() * (self.__nests[np.random.permutation(n)] - self.__nests[np.random.permutation(n)])
        self.__nests += np.multiply(stepsize, discovered.repeat(dimension).reshape(-1, dimension))
        self.__nests = np.clip(self.__nests, lb, ub)
        self.__nests[:, self.discrete] = np.round(self.__nests[:, self.discrete])
        return discovered

    def get_nests(self):
        """Return a history of cuckoos nests (return type: list)"""
        return self.__Nests
