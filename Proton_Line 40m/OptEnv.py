import get_beam_size
import numpy as np
import gym
from zoopt import Dimension, Parameter, Objective
from cpymad.madx import Madx


class kOptEvn(gym.Env):

    def __init__(self, q, s, o, a, solver, norm_vect, n_particles, _n_iter):
        self.dof = len(q) + len(s) + len(o) + len(a)
        self.rew = 100000000
        self.solver = solver
        self.q = q
        self.s = s
        self.o = o
        self.a = a
        # Collate all variables (quadrupoles, sextupoles, octupoles, distances)
        self.x0 = np.array(q + s + o + a)
        # Store actions, beam size, loss and fraction for every iteration
        self.x_all = np.zeros([1, self.dof])
        self.y_all = np.zeros([1, self.dof])
        self.x_best = np.zeros([1, self.dof])
        self.sigmax_all = np.zeros([1, 1])
        self.sigmay_all = np.zeros([1, 1])
        self.loss_all = np.zeros([1, 1])
        self.fraction_all = np.zeros([1, 1])
        # Vector to normalise actions
        self.norm_vect = norm_vect
        # Number of particles to track
        self.n_particles = n_particles
        # Max number of iterations
        self._n_iter = _n_iter
        # Spawn MAD-X process
        self.madx = Madx(stdout=False)
        self.madx.call(file='general_tt43_python.madx')
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)

        if solver == 'ZOOpt':
            dim = self.dof
            dimobj = Dimension(dim, [[-1, 1]] * dim, [True] * dim)
            self.parameter = Parameter(budget=self._n_iter,
                                       init_samples=[self.norm_data(self.x0)])
            self.step = Objective(self.step, dimobj)
        elif solver == 'BOBYQA':  # currently broken
            self.upper = np.ones([1, self.dof])
            self.lower = np.multiply(self.upper, -1)
        elif solver == 'Bayesian':  # currently unfinished
            dim = self.dof
            self.pbounds = {'x': [(-1, 1) * dim]}
        else:
            pass

    def render(self, mode='human'):
        pass

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, x_nor):

        if self.solver == 'ZOOpt':
            x_nor = x_nor.get_x()
        else:
            pass

        # print(x_nor) # normalised actions
        x_unnor = self.unnorm_data(x_nor) # normalise actions
        # print(x_unnor) # unnormalised actions

        c1 = get_beam_size.getBeamSize(x_unnor, self.n_particles, self.madx, len(self.q), len(self.s), len(self.o),
                                       len(self.a))
        a = (c1.get_beam_size())
        # where output is a = [beam_size_x, beam_size_y, beam_size_z, loss, percentage, error_flag]

        # Save values for current iteration
        self.sigmax_all = np.append(self.sigmax_all, a[0])
        self.sigmay_all = np.append(self.sigmay_all, a[1])
        self.y_all = np.append(self.y_all, [x_nor], axis=0)
        self.x_all = np.append(self.x_all, [x_unnor], axis=0)
        self.loss_all = np.append(self.loss_all, a[3])
        self.fraction_all = np.append(self.fraction_all, a[4])

        # If MAD-X has failed re-spawn process
        if a[5]:
            self.reset()

        print('Iteration' + str(len(self.loss_all)))
        print('Sigmax =' + str(a[0]) + ', Sigmay=' + str(a[1]))
        print('Total =' + str(a[0] ** 2 + a[1] ** 2))
        print('Percentage <5 um =' + str(a[4]))

        # Objective function with a = [beam_size_x, beam_size_y, beam_size_z, loss, percentage, error_flag, disp]
        # + 10*np.abs(a[2]-60.1)*np.abs(a[2]-60.1)
        output = (abs(a[0]) ** 2 + abs(a[1]) ** 2) * (1 + a[3]) * (1 + a[3]) + 100*a[6] + 100*a[7] + 0*(abs(a[1]-a[0]))**2
        print(output)
        print("xy_skew/kurtosis = " + str(a[6]))
        print("pxy_skew/kurtosis = " + str(a[7]))

        # print((a[0] ** 2 + a[1] ** 2) * (1 + a[3]) * (1 + a[3]))
        # If objective function is best so far, update x_best with new best parameters
        if output < self.rew:
            self.x_best = x_unnor
            self.rew = output
        print("best = " + str(self.rew))
        return output

    def norm_data(self, x_data):
        """
        Normalise the data
        """
        print(x_data)
        x_norm = np.divide(x_data, self.norm_vect)
        return x_norm

    def unnorm_data(self, x_norm):
        """
        Unnormalise the data
        """
        x_data = np.multiply(x_norm, self.norm_vect)
        return x_data

    def reset(self):
        """
         If MAD-X fails, re-spawn process
         """
        self.madx = Madx(stdout=False)
        self.madx.call(file='general_tt43_python.madx')
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
