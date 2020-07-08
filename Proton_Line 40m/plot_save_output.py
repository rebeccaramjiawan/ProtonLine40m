import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from scipy.interpolate import interp1d
from matplotlib import lines as mpl_lines
from matplotlib.ticker import FuncFormatter, MultipleLocator, FormatStrFormatter


class Plot:
    def __init__(self, madx):
        self.madx = madx

        params = {'axes.labelsize': 34,  # fontsize for x and y labels (was 10)
                  'axes.titlesize': 28,
                  'legend.fontsize': 28,  # was 10
                  'xtick.labelsize': 28,
                  'ytick.labelsize': 28,
                  'axes.linewidth': 1.5,
                  'lines.linewidth': 3,
                  'text.usetex': True,
                  'font.family': 'serif'
                  }
        matplotlib.rcParams.update(params)

    def twiss(self):
        """
        Twiss plots with synoptics
        """

        # print(twiss[1])

        # fig, ax = plt.subplots()
        fig = plt.figure()
        gs = matplotlib.gridspec.GridSpec(5, 1, height_ratios=[2, 0.25, 4, 0.25, 4])
        ax = fig.add_subplot(gs[2])
        ax1 = fig.add_subplot(gs[0])
        ay = fig.add_subplot(gs[4])
        self.madx.use(sequence='TT40TT41')
        self.madx.select(flag='twiss',
                         column=['name', 'keyword', 'k1l', 'k2l', 'k3l', 's', 'l', 'betx', 'bety', 'dx', 'dy', 'mux',
                                 'muy', 'env_x', 'env_y','aper_1', 'aper_2'])
        twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                x=0.00, px=0.00, y=0.0, py=0.0, deltap=str(-0.000),  file='test.txt')
        for idx in range(np.size(twiss['l'])):
            if twiss['aper_1'][idx] > 0:
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['S'][idx] - twiss['L'][idx], 0.2), twiss['L'][idx], -0.2 + twiss["aper_1"][idx]/2,
                        edgecolor='k', facecolor='w'))
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['S'][idx] - twiss['L'][idx], -0.2), twiss['L'][idx], 0.2 - twiss["aper_1"][idx]/2,
                        edgecolor='k', facecolor='w'))

        for idx in range(np.size(twiss['l'])):
            if twiss['aper_2'][idx] > 0:
                _ = ay.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['S'][idx] - twiss['L'][idx], 0.2), twiss['L'][idx], -0.2 + twiss["aper_2"][idx]/2,
                        edgecolor='k', facecolor='w'))
                _ = ay.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['S'][idx] - twiss['L'][idx], -0.2), twiss['L'][idx], 0.2 - twiss["aper_2"][idx]/2,
                        edgecolor='k', facecolor='w'))

        for idx in range(np.size(twiss['l'])):
            if twiss['keyword'][idx] == 'quadrupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], np.sign(twiss['k1l'][idx]),
                        facecolor='k', edgecolor='k'))
            elif twiss['keyword'][idx] == 'sextupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], 0.9 * np.sign(twiss['k2l'][idx]),
                        facecolor='b', edgecolor='b'))
            elif twiss['keyword'][idx] == 'octupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], 0.8 * np.sign(twiss['k3l'][idx]),
                        facecolor='r', edgecolor='r'))
            elif twiss['keyword'][idx] == 'rbend':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='g', edgecolor='g'))
            elif twiss['keyword'][idx] == 'vkicker':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='b', edgecolor='b'))
            elif twiss['keyword'][idx] == 'hkicker':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='b', edgecolor='b'))
            elif 'monitor' in twiss['keyword'][idx]:
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='w', edgecolor='r'))
            elif twiss['keyword'][idx] == 'instrument':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='w', edgecolor='r'))
        self.madx.select(flag='interpolate', step=0.05)
        self.madx.select(flag='twiss',
                         column=['name', 'keyword', 'k1l', 'k2l', 'k3l', 's', 'l', 'betx', 'bety', 'dx', 'dy', 'mux',
                                 'muy', 'env_x', 'env_y','aper_1', 'aper_2'])
        twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                x=0.00, px=0.00, y=0.0, py=0.0, deltap=str(-0.000))
        emitta = 8.21e-9
        DeltaP = 0.00035
        # xt = np.squeeze(np.array([np.sqrt(twiss['betx'] * emitta + (DeltaP * twiss['dx']) ** 2)]))
        # yt = np.squeeze(np.array([np.sqrt(twiss['bety'] * emitta + (DeltaP * twiss['dy']) ** 2)]))


        self.madx.input("xmax=self.madx.table.summ['betxmax'];")
        self.madx.input("ymax=self.madx.table.summ['betymax'];")
        self.madx.input("DeltaP = 0.00035;")
        self.madx.input("emitta = 8.21e-9;")
        self.madx.input("orbitError = 0.002;")
        self.madx.input("alignmentError = 0.002;")
        # self.madx.input(
        #     "sigma_x := 6*(sqrt((table(twiss, betx)*emitta) + (abs(table(twiss,dx))*DeltaP)*(abs(table(twiss,dx))*DeltaP)));")
        # self.madx.input(
        #     "sigma_y :=  6*(sqrt((table(twiss, bety)*emitta) + (abs(table(twiss,dy))*DeltaP)*(abs(table(twiss,dy))*DeltaP)));")
        self.madx.input(
            "sigma_x := 6*sqrt(table(twiss, betx)*emitta*1.1) + abs(table(twiss,dx))*DeltaP +0.001+0.001;")
        self.madx.input(
            "sigma_y := 6*sqrt(table(twiss, bety)*emitta*1.1) + abs(table(twiss,dy))* DeltaP +0.001+ 0.001;")
        self.madx.input("env_x := (6 * sqrt(emitta * 1.2 * table(twiss, betx)) + abs(table(twiss, dx)) * DeltaP + alignmentError + orbitError * sqrt(table(twiss, betx) / x_max));")
        self.madx.input(
            "env_y :=  (6 * sqrt(emitta * 1.2 * table(twiss, bety)) + abs(table(twiss, dy)) * DeltaP + alignmentError + orbitError * sqrt(table(twiss, bety) / y_max));")


        self.madx.use(sequence='TT40TT41')
        twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                x=0.00, px=0.00, y=0.0, py=0.0)

        ax.plot(twiss['s'], twiss['env_x'], 'r')
        ay.plot(twiss['s'], twiss['env_y'], 'g')
        ax.plot(twiss['s'], -twiss['env_x'], 'r')
        ay.plot(twiss['s'], -twiss['env_y'], 'g')
        # ax2 = ax.twinx()
        # ax2.plot(twiss['s'], twiss['Dx'], 'g')
        # ax2.plot(twiss['s'], twiss['Dy'], 'b')



        ay.set_xlabel("s [m]", fontsize=34, usetex=True, labelpad=10)
        ax.set_ylabel(r'env x [m]', fontsize=38, usetex=True, labelpad=10)
        ay.set_ylabel(r'env y [m]', fontsize=38, usetex=True, labelpad=10)

        # ax2.set_ylabel("$D_x, D_y$", fontsize=38, usetex=True, labelpad=10)
        ax.tick_params(labelsize=28)

        spine_color = 'gray'



        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set(xlim=(0, twiss['s'][-1]), ylim=(-0.05, 0.05))
        ay.set(xlim=(0, twiss['s'][-1]), ylim=(-0.05, 0.05))
        plt.gcf().subplots_adjust(bottom=0.15)
        ax1.set(xlim=(0, twiss['s'][-1]), ylim=(-1, 1))
        ax1.axes.get_yaxis().set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_position('center')

        for ax0 in [ax]:
            for spine in ['left', 'bottom', 'right']:
                ax0.spines[spine].set_color(spine_color)
                ax0.spines[spine].set_linewidth(2.5)
                ax0.tick_params(width=2.5, direction='out', color=spine_color)
                ax0.tick_params(labelsize=28, pad=10)
        plt.show()

    def diffTwiss(self):
        """
        Twiss plots with momentum offsets
        """

        fig, ax = plt.subplots()
        twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                x=0.00, px=0.00, y=0.0, py=0.0, deltap=str(-0.000))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k')
        ax.plot(twiss['s'], twiss['bety'], 'r')
        ax2 = ax.twinx()
        ax2.plot(twiss['s'], twiss['Dx'], 'g')
        ax2.plot(twiss['s'], twiss['Dy'], 'b')

        twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                x=0.00, px=0.00, y=0.0, py=0.0, deltap=str(-0.002))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k--')
        ax.plot(twiss['s'], twiss['bety'], 'r--')
        ax2.plot(twiss['s'], twiss['Dx'], 'g--')
        ax2.plot(twiss['s'], twiss['Dy'], 'b--')

        twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                x=0.00, px=0.00, y=0.0, py=0.0, deltap=str(0.002))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k--')
        ax.plot(twiss['s'], twiss['bety'], 'r--')

        ax2.plot(twiss['s'], twiss['Dx'], 'g--')
        ax2.plot(twiss['s'], twiss['Dy'], 'b--')

        ax.set_xlabel("s [m]", fontsize=34, usetex=True, labelpad=10)
        ax.set_ylabel(r'$\beta x, \beta y$ [m]', fontsize=38, usetex=True, labelpad=10)
        ax2.set_ylabel("$D_x, D_y$", fontsize=38, usetex=True, labelpad=10)
        ax.tick_params(labelsize=28)
        # ax.set_ylim = (-1, 1)
        spine_color = 'gray'

        ax.tick_params(labelsize=28, pad=10)
        ax2.tick_params(labelsize=28, pad=10)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for ax0 in [ax, ax2]:
            for spine in ['left', 'bottom', 'right']:
                ax0.spines[spine].set_color(spine_color)
                ax0.spines[spine].set_linewidth(2.5)
                ax0.tick_params(width=2.5, direction='out', color=spine_color)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('Chromatic effects')
        plt.show()

        fig, ax = plt.subplots()
        twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                px=0.00, y=0.0, py=0.0,
                                deltap=str(0), x=str(0))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k')
        ax.plot(twiss['s'], twiss['bety'], 'r')
        ax2 = ax.twinx()
        ax2.plot(twiss['s'], twiss['Dx'], 'g')
        ax2.plot(twiss['s'], twiss['Dy'], 'b')

        twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                deltap=str(0), x=str(-500e-6), px=str(-0.5e-4))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k--')
        ax.plot(twiss['s'], twiss['bety'], 'r--')
        ax2.plot(twiss['s'], twiss['Dx'], 'g--')
        ax2.plot(twiss['s'], twiss['Dy'], 'b--')

        twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                deltap=str(0), x=str(500e-6), px=str(0.5e-4))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k--')
        ax.plot(twiss['s'], twiss['bety'], 'r--')

        ax2.plot(twiss['s'], twiss['Dx'], 'g--')
        ax2.plot(twiss['s'], twiss['Dy'], 'b--')

        ax.set_xlabel("s [m]", fontsize=34, usetex=True, labelpad=10)
        ax.set_ylabel(r'$\beta x, \beta y$ [m]', fontsize=38, usetex=True, labelpad=10)
        ax2.set_ylabel("$D_x, D_y$", fontsize=38, usetex=True, labelpad=10)
        ax.tick_params(labelsize=28)
        # ax.set_ylim = (-1, 1)
        spine_color = 'gray'

        ax.tick_params(labelsize=28, pad=10)
        ax2.tick_params(labelsize=28, pad=10)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for ax0 in [ax, ax2]:
            for spine in ['left', 'bottom', 'right']:
                ax0.spines[spine].set_color(spine_color)
                ax0.spines[spine].set_linewidth(2.5)
                ax0.tick_params(width=2.5, direction='out', color=spine_color)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('Detuning with amplitude - x')
        plt.show()

        fig, ax = plt.subplots()
        twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                deltap=str(0), y=str(0))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k')
        ax.plot(twiss['s'], twiss['bety'], 'r')
        ax2 = ax.twinx()
        ax2.plot(twiss['s'], twiss['Dx'], 'g')
        ax2.plot(twiss['s'], twiss['Dy'], 'b')

        twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                deltap=str(0), y=str(-500e-6), py=str(-0.5e-4))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k--')
        ax.plot(twiss['s'], twiss['bety'], 'r--')
        ax2.plot(twiss['s'], twiss['Dx'], 'g--')
        ax2.plot(twiss['s'], twiss['Dy'], 'b--')

        twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                deltap=str(0), y=str(500e-6), py=str(0.5e-4))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k--')
        ax.plot(twiss['s'], twiss['bety'], 'r--')

        ax2.plot(twiss['s'], twiss['Dx'], 'g--')
        ax2.plot(twiss['s'], twiss['Dy'], 'b--')

        ax.set_xlabel("s [m]", fontsize=34, usetex=True, labelpad=10)
        ax.set_ylabel(r'$\beta x, \beta y$ [m]', fontsize=38, usetex=True, labelpad=10)
        ax2.set_ylabel("$D_x, D_y$", fontsize=38, usetex=True, labelpad=10)
        ax.tick_params(labelsize=28)
        # ax.set_ylim = (-1, 1)
        spine_color = 'gray'

        ax.tick_params(labelsize=28, pad=10)
        ax2.tick_params(labelsize=28, pad=10)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for ax0 in [ax, ax2]:
            for spine in ['left', 'bottom', 'right']:
                ax0.spines[spine].set_color(spine_color)
                ax0.spines[spine].set_linewidth(2.5)
                ax0.tick_params(width=2.5, direction='out', color=spine_color)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('Detuning with amplitude - y')
        plt.show()


    def survey(self):
        self.madx.use(sequence='TT40TT41')
        twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                x=0.00, px=0.00, y=0.0, py=0.0)

        self.madx.input("xmax=self.madx.table.summ['betxmax'];")
        self.madx.input("ymax=self.madx.table.summ['betymax'];")
        self.madx.input("DeltaP = 0.00035;")
        self.madx.input("emitta = 8.21e-9;")
        self.madx.input("orbitError = 0.001;")
        self.madx.input("alignmentError = 0.001;")
        # self.madx.input(
        #     "sigma_x := 6*(sqrt((table(twiss, betx)*emitta) + (abs(table(twiss,dx))*DeltaP)*(abs(table(twiss,dx))*DeltaP)));")
        # self.madx.input(
        #     "sigma_y :=  6*(sqrt((table(twiss, bety)*emitta) + (abs(table(twiss,dy))*DeltaP)*(abs(table(twiss,dy))*DeltaP)));")
        self.madx.input(
            "sigma_x:= 6*sqrt(emitta*1.2*table(twiss,betx))+abs(table(twiss,dx))* DeltaP +alignmentError+orbitError*sqrt(table(twiss,betx)/x_max);")
        self.madx.input(
            "sigma_y:= 6*sqrt(emitta*1.2*table(twiss,bety))+abs(table(twiss,dy))* DeltaP +alignmentError+ orbitError*sqrt(table(twiss,bety)/y_max);")

        self.madx.select(flag='twiss',
                         column=['name', 'keyword', 'k1l', 'k2l', 'k3l', 's', 'l', 'sigma_x', 'sigma_y', 'dx', 'dy', 'mux',
                                 'muy'])
        twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                x=0.00, px=0.00, y=0.0, py=0.0, file="test2.txt")


        survey = self.madx.survey(x0=-2314.71736,
                                  y0=2401.75556,
                                  z0=4456.10249,
                                  theta0=-2.17482033,
                                  phi0=0.000178,
                                  psi0=0)

        fig, ax = plt.subplots()
        plt.axis([4213, 4219, -3150, -2940])
        #plt.axis('equal')
        A = np.stack((twiss['s'], twiss['betx'], twiss['bety']), axis=1)
        A = np.sort(A, axis=0)
        # print(A)
        # ax.plot(twiss['s'], np.sqrt(twiss['betx']*6.2e-9), 'k--')
        # ax.plot(twiss['s'], np.sqrt(twiss['bety']*6.2e-9), 'r--')
        x = np.squeeze(np.array([-survey['z'][-1] + survey['z']]))
        y = np.squeeze(np.array([survey['x'][-1] - survey['x']]))

        emitta = 8.21e-9
        DeltaP = 0.00035
        xt = np.squeeze(twiss['sigma_x'])
        yt = np.squeeze(twiss['sigma_y'])
        theta = np.array(survey['theta'])

        ax.plot(survey['z'] + np.squeeze(np.multiply(1, xt, np.sin(theta))),
                survey['x'] + np.squeeze(np.multiply(1, xt, np.cos(theta))), 'r-', linewidth=1.3)
        ax.plot(survey['z'] - np.squeeze(np.multiply(1, xt, np.sin(theta))),
                survey['x'] - np.squeeze(np.multiply(1, xt, np.cos(theta))), 'r-', linewidth=1.3)

        ax.plot(survey['z'] + np.squeeze(np.multiply(1, xt, np.sin(theta))) - (0.001+(twiss['s'][twiss['name']=="plasma_merge:1"]-survey['s'])*0.0005) * np.sin(theta),
                survey['x'] + np.squeeze(np.multiply(1, xt, np.cos(theta))) - (0.001 +(twiss['s'][twiss['name']=="plasma_merge:1"]-survey['s'])*0.0005) *np.cos(theta), 'c-', linewidth=1.3)
        ax.plot(survey['z'] - np.squeeze(np.multiply(1, xt, np.sin(theta))) + (0.001+(twiss['s'][twiss['name']=="plasma_merge:1"]-survey['s'])*0.0005) *np.sin(theta),
                survey['x'] - np.squeeze(np.multiply(1, xt, np.cos(theta))) + (0.001 +(twiss['s'][twiss['name']=="plasma_merge:1"]-survey['s'])*0.0005) *np.cos(theta), 'c-', linewidth=1.3)
        # ax.plot(survey['z'] + np.squeeze(np.multiply(6, yt, np.sin(theta))),
        #         survey['x'] + np.squeeze(np.multiply(6, yt, np.cos(theta))), 'b-', linewidth=1)
        # ax.plot(survey['z'] - np.squeeze(np.multiply(6, yt, np.sin(theta))),
        #         survey['x'] - np.squeeze(np.msqueeze(np.multiply(6, xt, np.sin(theta))),
        #         survey['x'] + np.squeeze(np.multiply(6, xt, np.cos(theta))), 'r-', linewidth=1)
        # ax.plot(survey['z'] - np.squeeze(np.multiply(6, xt, np.sin(theta))),
        #         survey['x'] - np.squeeze(np.multiply(6, xt, np.cos(theta))), 'r-', linewidth=1)
        #
        # ax.plot(survey['z'] + np.squeeze(np.multiply(6, yt, np.sin(theta))),
        #         survey['x'] + np.squeeze(np.multiply(6, yt, np.cos(theta))), 'b-', linewidth=1)
        # ax.plot(survey['z'] - np.squeeze(np.multiply(6, yt, np.sin(theta))),
        #         survey['x'] - np.squeeze(np.multiply(6, yt, np.cos(theta))), 'b-', linewidth=1)

        for idx in range(np.size(survey['l'])):

            if twiss['name'][idx] == 'laser.1:1':
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (
                            survey['z'][idx] - survey['l'][idx] * np.cos(
                                (theta[idx] + theta[idx - 1]) / 2) + 0.1 * np.sin(
                                (theta[idx] + theta[idx - 1]) / 2),
                            survey['x'][idx] - 1 * np.cos((theta[idx] + theta[idx - 1]) / 2) - survey['l'][
                                idx] * np.sin(
                                (theta[idx] + theta[idx - 1]) / 2)), 0.2,
                        -survey['l'][idx],
                        angle=(np.pi / 2 + (theta[idx] + theta[idx - 1]) / 2) * 180 / np.pi, facecolor='m',
                        edgecolor='m'))
            elif twiss['name'][idx] == 'plasma.412402:1':
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (
                            survey['z'][idx] - survey['l'][idx] * np.cos(
                                (theta[idx] + theta[idx - 1]) / 2) + 0.25 * np.sin(
                                (theta[idx] + theta[idx - 1]) / 2),
                            survey['x'][idx] - 0.5 * np.cos((theta[idx] + theta[idx - 1]) / 2) - survey['l'][
                                idx] * np.sin(
                                (theta[idx] + theta[idx - 1]) / 2)), 0.5,
                        -survey['l'][idx],
                        angle=(np.pi / 2 + (theta[idx] + theta[idx - 1]) / 2) * 180 / np.pi, facecolor='m',
                        edgecolor='m'))
            elif twiss['name'][idx] == 'plasma.2:1':
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (
                            survey['z'][idx] - survey['l'][idx] * np.cos(
                                (theta[idx] + theta[idx - 1]) / 2) + 0.25 * np.sin(
                                (theta[idx] + theta[idx - 1]) / 2),
                            survey['x'][idx] - 0.5 * np.cos((theta[idx] + theta[idx - 1]) / 2) - survey['l'][
                                idx] * np.sin(
                                (theta[idx] + theta[idx - 1]) / 2)), 0.5,
                        -survey['l'][idx],
                        angle=(np.pi / 2 + (theta[idx] + theta[idx - 1]) / 2) * 180 / np.pi, facecolor='m',
                        edgecolor='m'))
            elif twiss['keyword'][idx] == 'quadrupole':
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (
                        survey['z'][idx] - survey['l'][idx] * np.cos((theta[idx] + theta[idx - 1]) / 2) + 0.25 * np.sin(
                            (theta[idx] + theta[idx - 1]) / 2),
                        survey['x'][idx] - 0.5 * np.cos((theta[idx] + theta[idx - 1]) / 2) - survey['l'][idx] * np.sin(
                            (theta[idx] + theta[idx - 1]) / 2)), 0.5,
                        -survey['l'][idx],
                        angle=(np.pi / 2 + (theta[idx] + theta[idx - 1]) / 2) * 180 / np.pi, facecolor='g',
                        edgecolor='g'))
            elif twiss['keyword'][idx] == 'rbend':
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (
                        survey['z'][idx] - survey['l'][idx] * np.cos((theta[idx] + theta[idx - 1]) / 2) + 0.25 * np.sin(
                            (theta[idx] + theta[idx - 1]) / 2),
                        survey['x'][idx] - 0.5 * np.cos((theta[idx] + theta[idx - 1]) / 2) - survey['l'][idx] * np.sin(
                            (theta[idx] + theta[idx - 1]) / 2)), 0.5,
                        -survey['l'][idx],
                        angle=(np.pi / 2 + (theta[idx] + theta[idx - 1]) / 2) * 180 / np.pi, facecolor='c',
                        edgecolor='c'))
            elif twiss['keyword'][idx] == 'vkicker':
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (
                        survey['z'][idx] - survey['l'][idx] * np.cos((theta[idx] + theta[idx - 1]) / 2) + 0.25 * np.sin(
                            (theta[idx] + theta[idx - 1]) / 2),
                        survey['x'][idx] - 0.5 * np.cos((theta[idx] + theta[idx - 1]) / 2) - survey['l'][idx] * np.sin(
                            (theta[idx] + theta[idx - 1]) / 2)), 0.5,
                        -survey['l'][idx],
                        angle=(np.pi / 2 + (theta[idx] + theta[idx - 1]) / 2) * 180 / np.pi, facecolor='b',
                        edgecolor='b'))
            elif twiss['keyword'][idx] == 'hkicker':
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (
                        survey['z'][idx] - survey['l'][idx] * np.cos((theta[idx] + theta[idx - 1]) / 2) + 0.25 * np.sin(
                            (theta[idx] + theta[idx - 1]) / 2),
                        survey['x'][idx] - 0.5 * np.cos((theta[idx] + theta[idx - 1]) / 2) - survey['l'][idx] * np.sin(
                            (theta[idx] + theta[idx - 1]) / 2)), 0.5,
                        -survey['l'][idx],
                        angle=(np.pi / 2 + (theta[idx] + theta[idx - 1]) / 2) * 180 / np.pi, facecolor='b',
                        edgecolor='b'))
            elif 'monitor' in twiss['keyword'][idx]:
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (
                        survey['z'][idx] - survey['l'][idx] * np.cos((theta[idx] + theta[idx - 1]) / 2) + 0.25 * np.sin(
                            (theta[idx] + theta[idx - 1]) / 2),
                        survey['x'][idx] - 0.5 * np.cos((theta[idx] + theta[idx - 1]) / 2) - survey['l'][idx] * np.sin(
                            (theta[idx] + theta[idx - 1]) / 2)), 0.5,
                        -survey['l'][idx],
                        angle=(np.pi / 2 + (theta[idx] + theta[idx - 1]) / 2) * 180 / np.pi, facecolor='r',
                        edgecolor='r'))
            elif twiss['keyword'][idx] == 'instrument':
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (
                        survey['z'][idx] - survey['l'][idx] * np.cos((theta[idx] + theta[idx - 1]) / 2) + 0.25 * np.sin(
                            (theta[idx] + theta[idx - 1]) / 2),
                        survey['x'][idx] - 0.5 * np.cos((theta[idx] + theta[idx - 1]) / 2) - survey['l'][idx] * np.sin(
                            (theta[idx] + theta[idx - 1]) / 2)), 0.5,
                        -survey['l'][idx],
                        angle=(np.pi / 2 + (theta[idx] + theta[idx - 1]) / 2) * 180 / np.pi, facecolor='r',
                        edgecolor='r'))

        self.plot_secant((survey['z'][twiss['name'][:] == "plasma.e:1"], survey['x'][twiss['name'][:] == "plasma.e:1"]),
                         (survey['z'][twiss['name'][:] == "table.1:1"], survey['x'][twiss['name'][:] == "table.1:1"]),
                         ax)
        ax.plot(survey['z'], survey['x'], 'k-', linewidth=1)
        ax.set_xlabel("z [m]", fontsize=34, usetex=True)
        ax.set_ylabel("x [m]", fontsize=34, usetex=True)
        self.madx.call(file='original_line.madx')
        self.madx.use(sequence='TT40TT41_original')

        twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                x=0.00, px=0.00, y=0.0, py=0.0)
        survey = self.madx.survey(x0=-2314.71736,
                                  y0=2401.75556,
                                  z0=4456.10249,
                                  theta0=-2.17482033,
                                  phi0=0.000178,
                                  psi0=0)
        ax.plot(survey['z'], survey['x'], 'g-', linewidth=1)
        # for idx in range(np.size(survey['l'])):
        #     if twiss['name'][idx] == 'plasma.412402:1':
        #         _ = ax.add_patch(
        #             matplotlib.patches.Rectangle(
        #                 (
        #                     survey['z'][idx] - survey['l'][idx] * np.cos((theta[idx] + theta[idx - 1]) / 2) + 0.25 * np.sin(
        #                         (theta[idx] + theta[idx - 1]) / 2),
        #                     survey['x'][idx] - 0.5 * np.cos((theta[idx] + theta[idx - 1]) / 2) - survey['l'][idx] * np.sin(
        #                         (theta[idx] + theta[idx - 1]) / 2)), 0.5,
        #                 -survey['l'][idx],
        #                 angle=(np.pi / 2 + (theta[idx] + theta[idx - 1]) / 2) * 180 / np.pi, facecolor='w',
        #                 edgecolor='m'))
        plt.show()

    def errors(self):

        no_seeds = 3000
        x1_off_all = np.zeros([0, 1])
        x2_off_all = np.zeros([0, 1])
        x_pl_off_all = np.zeros([0, 1])
        x_angle_off_all = np.zeros([0, 1])
        y1_off_all = np.zeros([0, 1])
        y2_off_all = np.zeros([0, 1])
        y_pl_off_all = np.zeros([0, 1])
        y_angle_off_all = np.zeros([0, 1])

        x1_corr_all = np.zeros([0, 1])
        x2_corr_all = np.zeros([0, 1])
        x_pl_corr_all = np.zeros([0, 1])
        x_angle_corr_all = np.zeros([0, 1])
        y1_corr_all = np.zeros([0, 1])
        y2_corr_all = np.zeros([0, 1])
        y_pl_corr_all = np.zeros([0, 1])
        y_angle_corr_all = np.zeros([0, 1])
        x1_corr_err_all = np.zeros([0, 1])
        x2_corr_err_all = np.zeros([0, 1])
        x_pl_corr_err_all = np.zeros([0, 1])
        x_angle_corr_err_all = np.zeros([0, 1])
        y1_corr_err_all = np.zeros([0, 1])
        y2_corr_err_all = np.zeros([0, 1])
        y_pl_corr_err_all = np.zeros([0, 1])
        y_angle_corr_err_all = np.zeros([0, 1])

        for ii in range(no_seeds):
            self.madx.input("ii =" + str(ii) + ";")
            self.madx.call(file='add_errors.madx')
            twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                    BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                    x=0.00, px=0.00, y=0.0, py=0.0, file="twiss_error.out")

            x1_off = twiss['x'][twiss['name'][:] == 'bpm.412352:1']
            x2_off = twiss['x'][twiss['name'][:] == 'bpm.412425:1']


            separation = twiss['s'][twiss['name'][:] == 'bpm.412352:1'] - twiss['s'][twiss['name'][:] == 'bpm.412425:1']
            x_angle_off = np.multiply(np.arctan((x2_off - x1_off) / separation), 1e6)

            x1_off_all = np.append(x1_off_all, np.multiply(x1_off, 1e6))
            x2_off_all = np.append(x2_off_all, np.multiply(x2_off, 1e6))
            x_pl_off = twiss['x'][twiss['name'][:] == 'plasma_merge:1']
            x_pl_off_all = np.append(x_pl_off_all, np.multiply(x_pl_off, 1e6))
            x_angle_off_all = np.append(x_angle_off_all, np.multiply(np.arctan((x2_off - x1_off) / separation), 1e6))

            y1_off = twiss['y'][twiss['name'][:] == 'bpm.412352:1']
            y2_off = twiss['y'][twiss['name'][:] == 'bpm.412425:1']
            y_angle_off = np.multiply(np.arctan((y2_off - y1_off) / separation), 1e6)

            y1_off_all = np.append(y1_off_all, np.multiply(y1_off, 1e6))
            y2_off_all = np.append(y2_off_all, np.multiply(y2_off, 1e6))
            y_angle_off_all = np.append(y_angle_off_all, np.multiply(np.arctan((y2_off - y1_off) / separation), 1e6))
            y_pl_off = twiss['y'][twiss['name'][:] == 'plasma_merge:1']
            y_pl_off_all = np.append(y_pl_off_all, np.multiply(y_pl_off, 1e6))
            self.madx.input(
                "CORRECT, flag = line, PLANE = x, MODE = svd, COND = 0, MONON = 1, MONERROR = 1, MONSCALE = 0, RESOUT = 0, ERROR = 1.E-5, CORRLIM = 10.0, clist = 'corr_x_valuesGlobal.tfs';")
            self.madx.input(
                "CORRECT, flag = line, PLANE = y, MODE = svd, COND = 0, MONON = 1, MONERROR = 1, MONSCALE = 0, RESOUT = 0, ERROR = 1.E-5, CORRLIM = 10.0, clist = 'corr_y_valuesGlobal.tfs';")

            twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                    BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                    x=0.00, px=0.00, y=0.0, py=0.0, file="twiss_error.out")
            x1_corr = twiss['x'][twiss['name'][:] == 'bpm.412352:1']
            x2_corr = twiss['x'][twiss['name'][:] == 'bpm.412425:1']
            x_angle_corr = np.multiply(np.arctan((x2_corr - x1_corr) / separation), 1e6)

            x1_corr_all = np.append(x1_corr_all, np.multiply(x1_corr, 1e6))
            x2_corr_all = np.append(x2_corr_all, np.multiply(x2_corr, 1e6))
            x_pl_corr = twiss['x'][twiss['name'][:] == 'plasma_merge:1']
            x_pl_corr_all = np.append(x_pl_corr_all, np.multiply(x_pl_corr, 1e6))
            x_angle_corr_all = np.append(x_angle_corr_all,
                                         np.multiply(np.arctan((x2_corr - x1_corr) / separation), 1e6))

            y1_corr = twiss['y'][twiss['name'][:] == 'bpm.412352:1']
            y2_corr = twiss['y'][twiss['name'][:] == 'bpm.412425:1']
            y_pl_corr = twiss['y'][twiss['name'][:] == 'plasma_merge:1']
            y_pl_corr_all = np.append(y_pl_corr_all, np.multiply(y_pl_corr, 1e6))
            y_angle_corr = np.multiply(np.arctan((y2_corr - y1_corr) / separation), 1e6)

            y1_corr_all = np.append(y1_corr_all, np.multiply(y1_corr, 1e6))
            y2_corr_all = np.append(y2_corr_all, np.multiply(y2_corr, 1e6))
            y_angle_corr_all = np.append(y_angle_corr_all,
                                         np.multiply(np.arctan((y2_corr - y1_corr) / separation), 1e6))

            self.madx.call(file='add_errors_jitt.madx')
            twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                    BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                    x=0.00, px=0.00, y=0.0, py=0.0, file="twiss_error.out")
            x1_corr_err = twiss['x'][twiss['name'][:] == 'bpm.412352:1']
            x2_corr_err = twiss['x'][twiss['name'][:] == 'bpm.412425:1']
            x_pl_corr_err = twiss['x'][twiss['name'][:] == 'plasma_merge:1']
            x_pl_corr_err_all = np.append(x_pl_corr_err_all, np.multiply(x_pl_corr_err, 1e6))
            x_angle_corr_err = np.multiply(np.arctan((x2_corr_err - x1_corr_err) / separation), 1e6)

            x1_corr_err_all = np.append(x1_corr_err_all, np.multiply(x1_corr_err, 1e6))
            x2_corr_err_all = np.append(x2_corr_err_all, np.multiply(x2_corr_err, 1e6))
            x_angle_corr_err_all = np.append(x_angle_corr_err_all,
                                             np.multiply(np.arctan((x2_corr_err - x1_corr_err) / separation), 1e6))

            y1_corr_err = twiss['y'][twiss['name'][:] == 'bpm.412352:1']
            y2_corr_err = twiss['y'][twiss['name'][:] == 'bpm.412425:1']
            y_angle_corr_err = np.multiply(np.arctan((y2_corr_err - y1_corr_err) / separation), 1e6)
            y_pl_corr_err = twiss['y'][twiss['name'][:] == 'plasma_merge:1']
            y_pl_corr_err_all = np.append(y_pl_corr_err_all, np.multiply(y_pl_corr_err, 1e6))
            y1_corr_err_all = np.append(y1_corr_err_all, np.multiply(y1_corr_err, 1e6))
            y2_corr_err_all = np.append(y2_corr_err_all, np.multiply(y2_corr_err, 1e6))
            y_angle_corr_err_all = np.append(y_angle_corr_err_all,
                                             np.multiply(np.arctan((y2_corr_err - x1_corr_err) / separation), 1e6))

        params = {'axes.labelsize': 34,  # fontsize for x and y labels (was 10)
                  'axes.titlesize': 28,
                  'legend.fontsize': 28,  # was 10
                  'xtick.labelsize': 28,
                  'ytick.labelsize': 28,
                  'axes.linewidth': 1.5,
                  'lines.linewidth': 3,
                  'text.usetex': True,
                  'font.family': 'serif'
                  }
        matplotlib.rcParams.update(params)
        plt.figure()
        plt.hist(x_angle_off_all, alpha=0.8, label="Errors (1$\sigma$ = %.2f $\mu$rad)" % np.std(x_angle_off_all))

        plt.hist(x_angle_corr_err_all, alpha=0.8,
                 label="+ Jitter (1$\sigma$ = %.2f $\mu$rad)" % np.std(x_angle_corr_err_all))
        plt.hist(x_angle_corr_all, alpha=0.8,
                 label="Corrected (1$\sigma$ = %.2f $\mu$rad)" % np.std(x_angle_corr_all))
        plt.legend()
        plt.xlabel('Angle in $\mu$rad')
        plt.ylabel('Frequency')
        plt.title('$x$ - pointing accuracy')
        plt.show()

        plt.figure()
        plt.hist(y_angle_off_all, alpha=0.8, label="Errors (1$\sigma$ = %.2f $\mu$rad)" % np.std(y_angle_off_all))

        plt.hist(y_angle_corr_err_all, alpha=0.8,
                 label="+ Jitter (1$\sigma$ = %.2f $\mu$rad)" % np.std(y_angle_corr_err_all))
        plt.hist(y_angle_corr_all, alpha=0.8,
                 label="Corrected (1$\sigma$ = %.2f $\mu$rad)" % np.std(y_angle_corr_all))
        plt.legend()
        plt.xlabel('Angle in $\mu$rad')
        plt.ylabel('Frequency')
        plt.title('$y$ - pointing accuracy')
        plt.show()

        plt.figure()
        plt.hist(x_pl_off_all, alpha=0.8, label="Errors (1$\sigma_x$ = %.2f $\mu$m)" % np.std(x_pl_off_all))

        plt.hist(x_pl_corr_err_all, alpha=0.8,
                 label="+ Jitter (1$\sigma_x$ = %.2f $\mu$m)" % np.std(x_pl_corr_err_all))
        plt.hist(x_pl_corr_all, alpha=0.8,
                 label="Corrected (1$\sigma_x$ = %.2f $\mu$m)" % np.std(x_pl_corr_all))
        plt.legend()
        plt.xlabel('Position in $\mu$m')
        plt.ylabel('Frequency')
        plt.title('$x$ - position at merge-point')
        plt.show()
        plt.figure()
        plt.hist(y_pl_off_all, alpha=0.8, label="Errors (1$\sigma_y$ = %.2f $\mu$m)" % np.std(y_pl_off_all))

        plt.hist(y_pl_corr_err_all, alpha=0.8,
                 label="+ Jitter (1$\sigma_y$ = %.2f $\mu$m)" % np.std(y_pl_corr_err_all))
        plt.hist(y_pl_corr_all, alpha=0.8,
                 label="Corrected (1$\sigma_y$ = %.2f $\mu$m)" % np.std(y_pl_corr_all))
        plt.legend()
        plt.xlabel('Position in $\mu$m')
        plt.ylabel('Frequency')
        plt.title('$y$ - position at merge-point')
        plt.show()

    def slope_from_points(self, point1, point2):
        return (point2[1] - point1[1]) / (point2[0] - point1[0])

    def plot_secant(self, point1, point2, ax):
        # plot the secant
        slope = self.slope_from_points(point1, point2)
        intercept = point1[1] - slope * point1[0]
        # update the points to be on the axes limits
        x = ax.get_xlim()
        y = ax.get_ylim()
        data_y = [x[0] * slope + intercept, x[1] * slope + intercept]
        line = mpl_lines.Line2D(x, data_y, color='b', linewidth=1.2)
        ax.add_line(line)

    def plot_phase_advance(self):
        # self.madx.use(sequence='TT43')
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
        twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                x=0.00, px=0.00, y=0.0, py=0.0, deltap=str(-0.000))
        fig = plt.figure()
        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        ax = fig.add_subplot(gs[1])
        ax1 = fig.add_subplot(gs[0])
        ax.plot(twiss['s'][:], np.multiply(twiss['mux'][:], 2 ), label='$\mu_x$')
        ax.plot(twiss['s'][:], np.multiply(twiss['muy'][:], 2 ), label='$\mu_y$')
        for idx in range(np.size(twiss['l'])):
            if twiss['name'][idx] == 'plasma_merge:1':
                ax.plot(twiss['s'][idx], np.multiply(twiss['mux'][idx], 2 ), 'gx', markersize='15',
                        label='plasma merge')
                ax.plot(twiss['s'][idx], np.multiply(twiss['muy'][idx], 2 ), 'gx', markersize='15')
                # ax.annotate('(%s, %s)' % xy, xy= ((twiss['s'][idx]), (np.multiply(twiss['muy'][idx], 2 ))), textcoords='data')
                # ax.annotate('%.3f' % round(np.multiply(twiss['muy'][idx], 2 ),5),
                #     xy=(twiss['s'][idx], (np.multiply(twiss['muy'][idx], 2 ))),  # theta, radius
                #     xytext=((twiss['s'][idx])*1.008, (np.multiply(twiss['muy'][idx], 2 ))),size=20)
                # ax.annotate('%.3f' % round(np.multiply(twiss['mux'][idx], 2 ), 5),
                #             xy=(twiss['s'][idx], (np.multiply(twiss['mux'][idx], 2 ))),  # theta, radius
                #             xytext=((twiss['s'][idx]) * 1.008, (np.multiply(twiss['mux'][idx], 2 ))), size=20)
                plasma_mux = np.multiply(twiss['mux'][idx], 2)
                plasma_muy = np.multiply(twiss['muy'][idx], 2)
        sign = -1
        for idx in range(np.size(twiss['l'])):
            if 'btv' in twiss['name'][idx]:
                sign = -sign
                ax.plot(twiss['s'][idx], np.multiply(twiss['mux'][idx], 2 ), 'rx', markersize='15')
                ax.plot(twiss['s'][idx], np.multiply(twiss['muy'][idx], 2 ), 'rx', markersize='15')
                # ax.annotate('(%s, %s)' % xy, xy= ((twiss['s'][idx]), (np.multiply(twiss['muy'][idx], 2 ))), textcoords='data')
                ax.annotate('%.3f $\pi$' % round(np.multiply(twiss['muy'][idx], 2) - plasma_muy, 5),
                            xy=(twiss['s'][idx], (np.multiply(twiss['muy'][idx], 2 ))),  # theta, radius
                            xytext=((twiss['s'][idx]) * 0.995,
                                    (1 + 0.02 * sign) * (np.multiply(twiss['muy'][idx], 2 ))),
                            size=20)
                ax.annotate('%.3f $\pi$' % round(np.multiply(twiss['mux'][idx], 2) - plasma_mux, 5),
                            xy=(twiss['s'][idx], (np.multiply(twiss['mux'][idx], 2 ))),  # theta, radius
                            xytext=((twiss['s'][idx]) * 0.995,
                                    (1 + 0.02 * sign) * (np.multiply(twiss['mux'][idx], 2 ))),
                            size=20)
        # for idx in range(np.size(twiss['l'])):
        #     if 'bpm' in twiss['name'][idx]:
        #         ax.plot(twiss['s'][idx], np.multiply(twiss['mux'][idx], 2 ), 'bx', markersize='15')
        #         ax.plot(twiss['s'][idx], np.multiply(twiss['muy'][idx], 2 ), 'bx', markersize='15')
        #         # ax.annotate('(%s, %s)' % xy, xy= ((twiss['s'][idx]), (np.multiply(twiss['muy'][idx], 2 ))), textcoords='data')
        #         ax.annotate('%.3f' % round(np.multiply(twiss['muy'][idx], 2 )-plasma_mux, 5),
        #                     xy=(twiss['s'][idx], (np.multiply(twiss['muy'][idx], 2 ))),  # theta, radius
        #                     xytext=((twiss['s'][idx])*1.01, 0.99*(np.multiply(twiss['muy'][idx], 2 ))), size=20, color = 'b')
        #         ax.annotate('%.3f' % round(np.multiply(twiss['mux'][idx], 2 )-plasma_muy, 5),
        #                     xy=(twiss['s'][idx], (np.multiply(twiss['mux'][idx], 2 ))),  # theta, radius
        #                     xytext=((twiss['s'][idx]) * 0.99, 1.01*(np.multiply(twiss['mux'][idx], 2 ))), size=20, color ='b')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g $\pi$'))

        for idx in range(np.size(twiss['l'])):
            print(twiss['name'][idx])
            if twiss['name'][idx] == 'plasma.412402:1':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='m', edgecolor='m'))

            elif twiss['name'][idx] == 'plasma.2:1':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='m', edgecolor='m'))
            elif twiss['name'][idx] == 'table.1:1':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='m', edgecolor='m'))
            elif twiss['keyword'][idx] == 'quadrupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], np.sign(twiss['k1l'][idx]),
                        facecolor='k', edgecolor='k'))
            elif twiss['keyword'][idx] == 'sextupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], 0.9 * np.sign(twiss['k2l'][idx]),
                        facecolor='b', edgecolor='b'))
            elif twiss['keyword'][idx] == 'octupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], 0.8 * np.sign(twiss['k3l'][idx]),
                        facecolor='r', edgecolor='r'))
            elif twiss['keyword'][idx] == 'rbend':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='g', edgecolor='g'))
            elif twiss['keyword'][idx] == 'vkicker':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='b', edgecolor='b'))
            elif twiss['keyword'][idx] == 'hkicker':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='b', edgecolor='b'))
            elif 'monitor' in twiss['keyword'][idx]:
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='w', edgecolor='r'))
            elif twiss['keyword'][idx] == 'instrument':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='w', edgecolor='r'))

        plt.gcf().subplots_adjust(bottom=0.15)
        ax1.set(xlim=(0, twiss['s'][-1]), ylim=(-1, 1))
        ax1.axes.get_yaxis().set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_position('center')

        ax.legend()
        ax.set_xlabel('s [m]')
        ax.set_ylabel('Phase advance [rad]')
        ax.set_xlim(650, 860)
        ax1.set_xlim(650, 860)
        ax.set_ylim(15/np.pi, 22/np.pi)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1))
        plt.title('btvs')
        plt.show()

        fig = plt.figure()
        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        ax = fig.add_subplot(gs[1])
        ax1 = fig.add_subplot(gs[0])
        ax.plot(twiss['s'][:], np.multiply(twiss['mux'][:], 2), label='$\mu_x$')
        ax.plot(twiss['s'][:], np.multiply(twiss['muy'][:], 2), label='$\mu_y$')
        for idx in range(np.size(twiss['l'])):
            if twiss['name'][idx] == 'plasma_merge:1':
                ax.plot(twiss['s'][idx], np.multiply(twiss['mux'][idx], 2 ), 'bx', markersize='15',
                        label='plasma merge')
                ax.plot(twiss['s'][idx], np.multiply(twiss['muy'][idx], 2 ), 'bx', markersize='15')
                # ax.annotate('(%s, %s)' % xy, xy= ((twiss['s'][idx]), (np.multiply(twiss['muy'][idx], 2 ))), textcoords='data')
                # ax.annotate('%.3f' % round(np.multiply(twiss['muy'][idx], 2 ),5),
                #     xy=(twiss['s'][idx], (np.multiply(twiss['muy'][idx], 2 ))),  # theta, radius
                #     xytext=((twiss['s'][idx])*1.008, (np.multiply(twiss['muy'][idx], 2 ))),size=20)
                # ax.annotate('%.3f' % round(np.multiply(twiss['mux'][idx], 2 ), 5),
                #             xy=(twiss['s'][idx], (np.multiply(twiss['mux'][idx], 2 ))),  # theta, radius
                #             xytext=((twiss['s'][idx]) * 1.008, (np.multiply(twiss['mux'][idx], 2 ))), size=20)
                plasma_mux = np.multiply(twiss['mux'][idx], 2)
                plasma_muy = np.multiply(twiss['muy'][idx], 2)
        sign = 1
        for idx in range(np.size(twiss['l'])):
            if 'bpm' in twiss['name'][idx]:
                sign = -sign
                ax.plot(twiss['s'][idx], np.multiply(twiss['mux'][idx], 2 ), 'rx', markersize='15')
                ax.plot(twiss['s'][idx], np.multiply(twiss['muy'][idx], 2 ), 'rx', markersize='15')
                # ax.annotate('(%s, %s)' % xy, xy= ((twiss['s'][idx]), (np.multiply(twiss['muy'][idx], 2 ))), textcoords='data')
                ax.annotate('%.3f $\pi$' % round(np.multiply(twiss['muy'][idx], 2) - plasma_muy, 5),
                            xy=(twiss['s'][idx], (np.multiply(twiss['muy'][idx], 2 ))),  # theta, radius
                            xytext=((twiss['s'][idx]) * 0.995, (1+0.02*sign) * (np.multiply(twiss['muy'][idx], 2 ))),
                            size=20)
                ax.annotate('%.3f $\pi$' % round(np.multiply(twiss['mux'][idx], 2) - plasma_mux, 5),
                            xy=(twiss['s'][idx], (np.multiply(twiss['mux'][idx], 2 ))),  # theta, radius
                            xytext=((twiss['s'][idx]) * 0.995, (1+0.02*sign) * (np.multiply(twiss['mux'][idx], 2))),
                            size=20)
        # for idx in range(np.size(twiss['l'])):
        #     if 'bpm' in twiss['name'][idx]:
        #         ax.plot(twiss['s'][idx], np.multiply(twiss['mux'][idx], 2 ), 'bx', markersize='15')
        #         ax.plot(twiss['s'][idx], np.multiply(twiss['muy'][idx], 2 ), 'bx', markersize='15')
        #         # ax.annotate('(%s, %s)' % xy, xy= ((twiss['s'][idx]), (np.multiply(twiss['muy'][idx], 2 ))), textcoords='data')
        #         ax.annotate('%.3f' % round(np.multiply(twiss['muy'][idx], 2 )-plasma_mux, 5),
        #                     xy=(twiss['s'][idx], (np.multiply(twiss['muy'][idx], 2 ))),  # theta, radius
        #                     xytext=((twiss['s'][idx])*1.01, 0.99*(np.multiply(twiss['muy'][idx], 2 ))), size=20, color = 'b')
        #         ax.annotate('%.3f' % round(np.multiply(twiss['mux'][idx], 2 )-plasma_muy, 5),
        #                     xy=(twiss['s'][idx], (np.multiply(twiss['mux'][idx], 2 ))),  # theta, radius
        #                     xytext=((twiss['s'][idx]) * 0.99, 1.01*(np.multiply(twiss['mux'][idx], 2 ))), size=20, color ='b')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g $\pi$'))

        for idx in range(np.size(twiss['l'])):
            if twiss['name'][idx] == 'plasma.412402:1':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='m', edgecolor='m'))
            elif twiss['name'][idx] == 'table.1:1':
                    _ = ax1.add_patch(
                        matplotlib.patches.Rectangle(
                            (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                            facecolor='m', edgecolor='m'))
            elif twiss['keyword'][idx] == 'quadrupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], np.sign(twiss['k1l'][idx]),
                        facecolor='k', edgecolor='k'))
            elif twiss['keyword'][idx] == 'sextupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], 0.9 * np.sign(twiss['k2l'][idx]),
                        facecolor='b', edgecolor='b'))
            elif twiss['keyword'][idx] == 'octupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], 0.8 * np.sign(twiss['k3l'][idx]),
                        facecolor='r', edgecolor='r'))
            elif twiss['keyword'][idx] == 'rbend':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='g', edgecolor='g'))
            elif twiss['keyword'][idx] == 'vkicker':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='b', edgecolor='b'))
            elif twiss['keyword'][idx] == 'hkicker':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='b', edgecolor='b'))
            elif 'monitor' in twiss['keyword'][idx]:
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='w', edgecolor='r'))
            elif twiss['keyword'][idx] == 'instrument':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='w', edgecolor='r'))

        plt.gcf().subplots_adjust(bottom=0.15)
        ax1.set(xlim=(0, twiss['s'][-1]), ylim=(-1, 1))
        ax1.axes.get_yaxis().set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_position('center')

        ax.legend()
        ax.set_xlabel('s [m]')
        ax.set_ylabel('Phase advance [rad]')
        ax.set_xlim(650, 860)
        ax1.set_xlim(650, 860)
        ax.set_ylim(15/np.pi, 22/np.pi)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1))
        plt.title('bpms')
        plt.show()


        fig = plt.figure()
        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        ax = fig.add_subplot(gs[1])
        ax1 = fig.add_subplot(gs[0])
        ax.plot(twiss['s'][:], np.multiply(twiss['mux'][:], 2 ), label='$\mu_x$')
        ax.plot(twiss['s'][:], np.multiply(twiss['muy'][:], 2 ), label='$\mu_y$')
        for idx in range(np.size(twiss['l'])):
            if twiss['name'][idx] == 'plasma_merge:1':
                ax.plot(twiss['s'][idx], np.multiply(twiss['mux'][idx], 2), 'gx', markersize='15',
                        label='plasma merge')
                ax.plot(twiss['s'][idx], np.multiply(twiss['muy'][idx], 2 ), 'gx', markersize='15')
                # ax.annotate('(%s, %s)' % xy, xy= ((twiss['s'][idx]), (np.multiply(twiss['muy'][idx], 2 ))), textcoords='data')
                # ax.annotate('%.3f' % round(np.multiply(twiss['muy'][idx], 2 ),5),
                #     xy=(twiss['s'][idx], (np.multiply(twiss['muy'][idx], 2 ))),  # theta, radius
                #     xytext=((twiss['s'][idx])*1.008, (np.multiply(twiss['muy'][idx], 2 ))),size=20)
                # ax.annotate('%.3f' % round(np.multiply(twiss['mux'][idx], 2 ), 5),
                #             xy=(twiss['s'][idx], (np.multiply(twiss['mux'][idx], 2 ))),  # theta, radius
                #             xytext=((twiss['s'][idx]) * 1.008, (np.multiply(twiss['mux'][idx], 2 ))), size=20)
                plasma_mux = np.multiply(twiss['mux'][idx], 2)
                plasma_muy = np.multiply(twiss['muy'][idx], 2)
        for idx in range(np.size(twiss['l'])):
            if 'vkick' in twiss['keyword'][idx]:
                ax.plot(twiss['s'][idx], np.multiply(twiss['mux'][idx], 2 ), 'rx', markersize='15')
                ax.plot(twiss['s'][idx], np.multiply(twiss['muy'][idx], 2 ), 'rx', markersize='15')
                # ax.annotate('(%s, %s)' % xy, xy= ((twiss['s'][idx]), (np.multiply(twiss['muy'][idx], 2 ))), textcoords='data')
                ax.annotate('%.3f $\pi$' % round(np.multiply(twiss['muy'][idx], 2) - plasma_muy, 5),
                            xy=(twiss['s'][idx], (np.multiply(twiss['muy'][idx], 2 ))),  # theta, radius
                            xytext=((twiss['s'][idx]), 0.975 * (np.multiply(twiss['muy'][idx], 2 ))),
                            size=20, color='r')
                ax.annotate('%.3f $\pi$' % round(np.multiply(twiss['mux'][idx], 2) - plasma_mux, 5),
                            xy=(twiss['s'][idx], (np.multiply(twiss['mux'][idx], 2))),  # theta, radius
                            xytext=((twiss['s'][idx]), 0.975 * (np.multiply(twiss['mux'][idx], 2 ))),
                            size=20, color='r')
            if 'hkick' in twiss['keyword'][idx]:
                ax.plot(twiss['s'][idx], np.multiply(twiss['mux'][idx], 2 ), 'gx', markersize='15')
                ax.plot(twiss['s'][idx], np.multiply(twiss['muy'][idx], 2 ), 'gx', markersize='15')
                # ax.annotate('(%s, %s)' % xy, xy= ((twiss['s'][idx]), (np.multiply(twiss['muy'][idx], 2 ))), textcoords='data')
                ax.annotate('%.3f $\pi$' % round(np.multiply(twiss['muy'][idx], 2) - plasma_muy, 5),
                            xy=(twiss['s'][idx], (np.multiply(twiss['muy'][idx], 2 ))),  # theta, radius
                            xytext=((twiss['s'][idx]), 1.025 * (np.multiply(twiss['muy'][idx], 2 ))),
                            size=20, color='g')
                ax.annotate('%.3f $\pi$' % round(np.multiply(twiss['mux'][idx], 2) - plasma_mux, 5),
                            xy=(twiss['s'][idx], (np.multiply(twiss['mux'][idx], 2))),  # theta, radius
                            xytext=((twiss['s'][idx]) , 1.025 * (np.multiply(twiss['mux'][idx], 2 ))),
                            size=20, color='g')
        # for idx in range(np.size(twiss['l'])):
        #     if 'bpm' in twiss['name'][idx]:
        #         ax.plot(twiss['s'][idx], np.multiply(twiss['mux'][idx], 2 ), 'bx', markersize='15')
        #         ax.plot(twiss['s'][idx], np.multiply(twiss['muy'][idx], 2 ), 'bx', markersize='15')
        #         # ax.annotate('(%s, %s)' % xy, xy= ((twiss['s'][idx]), (np.multiply(twiss['muy'][idx], 2 ))), textcoords='data')
        #         ax.annotate('%.3f' % round(np.multiply(twiss['muy'][idx], 2 )-plasma_mux, 5),
        #                     xy=(twiss['s'][idx], (np.multiply(twiss['muy'][idx], 2 ))),  # theta, radius
        #                     xytext=((twiss['s'][idx])*1.01, 0.99*(np.multiply(twiss['muy'][idx], 2 ))), size=20, color = 'b')
        #         ax.annotate('%.3f' % round(np.multiply(twiss['mux'][idx], 2 )-plasma_muy, 5),
        #                     xy=(twiss['s'][idx], (np.multiply(twiss['mux'][idx], 2 ))),  # theta, radius
        #                     xytext=((twiss['s'][idx]) * 0.99, 1.01*(np.multiply(twiss['mux'][idx], 2 ))), size=20, color ='b')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g $\pi$'))

        for idx in range(np.size(twiss['l'])):
            if twiss['name'][idx] == 'plasma.412402:1':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='m', edgecolor='m'))
            elif twiss['name'][idx] == 'table.1:1':
                    _ = ax1.add_patch(
                        matplotlib.patches.Rectangle(
                            (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                            facecolor='m', edgecolor='m'))
            elif twiss['keyword'][idx] == 'quadrupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], np.sign(twiss['k1l'][idx]),
                        facecolor='k', edgecolor='k'))
            elif twiss['keyword'][idx] == 'sextupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], 0.9 * np.sign(twiss['k2l'][idx]),
                        facecolor='b', edgecolor='b'))
            elif twiss['keyword'][idx] == 'octupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], 0.8 * np.sign(twiss['k3l'][idx]),
                        facecolor='r', edgecolor='r'))
            elif twiss['keyword'][idx] == 'rbend':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='g', edgecolor='g'))
            elif twiss['keyword'][idx] == 'vkicker':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='b', edgecolor='b'))
            elif twiss['keyword'][idx] == 'hkicker':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='b', edgecolor='b'))
            elif 'monitor' in twiss['keyword'][idx]:
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='w', edgecolor='r'))
            elif twiss['keyword'][idx] == 'instrument':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='w', edgecolor='r'))

        plt.gcf().subplots_adjust(bottom=0.15)
        ax1.set(xlim=(0, twiss['s'][-1]), ylim=(-1, 1))
        ax1.axes.get_yaxis().set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_position('center')

        ax.legend()
        ax.set_xlabel('s [m]')
        ax.set_ylabel('Phase advance [rad]')
        ax.set_xlim(650, 860)
        ax1.set_xlim(650, 860)
        ax.set_ylim(15/np.pi, 22/np.pi)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1))
        plt.title('correctors')
        plt.show()

    def offset(self):
        fig = plt.figure()
        plt.title("Plasma cell shifted by 40 m from Run 1 position")
        ax = fig.add_subplot()
        # gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        # ax = fig.add_subplot(gs[1])
        # ax1 = fig.add_subplot(gs[0])
        for i in range(1):
            # angle = [0.6289, 0.662, 0.6951, 0.7283, 0.7615]
            # mbg = [0, 0.5, 1, 1.5, 2]

            mbg = [5]
            color = ['c']
            legend = ['5 m']

            self.madx.use(sequence='TT40TT41')
            self.madx.select(flag='twiss',
                             column=['name', 'keyword', 'k1l', 'k2l', 'k3l', 's', 'l', 'sigma_x', 'sigma_y', 'dx', 'dy', 'mux',
                                     'muy', 'angle'])
            twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                    BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                    x=0.00, px=0.00, y=0.0, py=0.0)
            angle = -twiss['angle'][twiss["name"] == "mbhfd.412133:1"]
            self.madx.input("b12_temp = %f;" % angle[i])
            self.madx.input("xmax=self.madx.table.summ['betxmax'];")
            self.madx.input("ymax=self.madx.table.summ['betymax'];")
            self.madx.input("DeltaP = 0.00035;")
            self.madx.input("emitta = 8.21e-9;")
            self.madx.input("orbitError = 0.001;")
            self.madx.input("alignmentError = 0.001;")
            self.madx.input(
                "sigma_x:= 6*sqrt(emitta*1.2*table(twiss,betx))+abs(table(twiss,dx))* DeltaP +alignmentError+orbitError*sqrt(table(twiss,betx)/x_max);")
            self.madx.input(
                "sigma_y:= 6*sqrt(emitta*1.2*table(twiss,bety))+abs(table(twiss,dy))* DeltaP +alignmentError+ orbitError*sqrt(table(twiss,bety)/y_max);")

            self.madx.select(flag='twiss',
                             column=['name', 'keyword', 'k1l', 'k2l', 'k3l', 's', 'l', 'sigma_x', 'sigma_y', 'dx', 'dy', 'mux',
                                     'muy'])
            twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                    BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                    x=0.00, px=0.00, y=0.0, py=0.0)


            survey = self.madx.survey(x0=-2314.71736,
                                      y0=2401.75556,
                                      z0=4456.10249,
                                      theta0=-2.17482033,
                                      phi0=0.000178,
                                      psi0=0)

            # fig, ax = plt.subplots()


            xt = np.squeeze(twiss['sigma_x'])
            yt = np.squeeze(twiss['sigma_y'])
            theta = np.array(survey['theta'])
            chicane1 = np.zeros(np.shape(twiss['s']))
            chicane2 = np.zeros(np.shape(twiss['s']))
            chicane3 = np.zeros(np.shape(twiss['s']))
            chicane4 = np.zeros(np.shape(twiss['s']))
            chicane5 = np.zeros(np.shape(twiss['s']))

            for idx in range(np.size(twiss['s'])):
                if twiss['s'][idx] < twiss['s'][twiss['name'][:] == "mbhfd.412330:1"]:
                    chicane1[idx] = np.sin(angle[i])*(-twiss['s'][idx]+twiss['s'][twiss['name'][:] == "mbhfd.412330:1"]-0.95)
            for idx in range(np.size(twiss['s'])):
                if twiss['s'][idx] < twiss['s'][twiss['name'][:] == "mbhfd.412324:1"]:
                    chicane2[idx] = np.sin(angle[i])*(-twiss['s'][idx]+twiss['s'][twiss['name'][:] == "mbhfd.412324:1"]-0.95)
            for idx in range(np.size(twiss['s'])):
                if twiss['s'][idx] < twiss['s'][twiss['name'][:] == "mbhfd.412141:1"]:
                    chicane3[idx] = np.sin(-angle[i])*(-twiss['s'][idx]+twiss['s'][twiss['name'][:] == "mbhfd.412141:1"]-0.95)
            for idx in range(np.size(twiss['s'])):
                if twiss['s'][idx] < twiss['s'][twiss['name'][:] == "mbhfd.412133:1"]:
                    chicane4[idx] = np.sin(-angle[i])*(-twiss['s'][idx]+twiss['s'][twiss['name'][:] == "mbhfd.412133:1"]-0.95)
            for idx in range(np.size(twiss['s'])):
                if twiss['s'][idx] < twiss['s'][twiss['name'][:] ==  "mbg.412115:1"]:
                    chicane5[idx] = np.sin(-0.00799999)*(-twiss['s'][idx]+twiss['s'][twiss['name'][:] == "mbg.412115:1"]-6.3/2)

            total_chicane = -chicane1 - chicane2 -chicane3 - chicane4- chicane5
            total_angle = np.flip(twiss['angle'][:])

            total_angle = (np.flip(np.cumsum(total_angle)))
            ax.plot(twiss['s'], total_chicane + np.multiply(xt, 1), color[i], label='%s' % legend[i], lw=2)
            ax.plot(twiss['s'], total_chicane - np.multiply(xt, 1), color[i], lw=2)
            # ax.plot(twiss['s'], total_chicane , 'kx-', lw=2)

            ax.fill_between(twiss['s'], total_chicane + np.multiply(xt, 1), total_chicane - np.multiply(xt, 1), color='lightcyan')
            ax.text(766.5, -0.18, '6$\sigma$ proton beam env.', color='darkcyan', fontsize=23)
            ax.text(825, 0.015, 'laser beam', color='m', fontsize=23)
            ax.text(816, -0.005, 'laser mirror', color='m', fontsize=23)
            ax.text(839, -0.022, 'plasma cell', color='chocolate', fontsize=23)
            after_laser = (twiss['s'][twiss['name'] == 'laser.1:1'] <= twiss['s'])
            before_merge = (twiss['s'] <= twiss['s'][twiss['name'] == 'plasma_merge:1'])
            laser_exist = np.logical_and(after_laser, before_merge)

            ax.plot(twiss['s'][laser_exist], (0.001 + (twiss['s'][twiss['name'] == "plasma_merge:1"] - twiss['s'][laser_exist]) * np.sin(0.0005)), 'm', lw=1.5)
            ax.plot(twiss['s'][laser_exist], (-0.001 - (twiss['s'][twiss['name'] == "plasma_merge:1"] - twiss['s'][laser_exist]) * np.sin(0.0005)), 'm', lw=1.5)
            ax.fill_between(twiss['s'][laser_exist], (0.001 + (twiss['s'][twiss['name'] == "plasma_merge:1"] - twiss['s'][laser_exist]) * np.sin(0.0005)), (-0.001 - (twiss['s'][twiss['name'] == "plasma_merge:1"] - twiss['s'][laser_exist]) * np.sin(0.0005)), color="thistle")


            ax.set_xlabel('Longitudinal position $s$ [m]')
            ax.set_ylabel('Horizontal position $x$ [m]')

            self.madx.select(flag='twiss',
                             column=['name', 'keyword', 'k1l', 'k2l', 'k3l', 's', 'l', 'betx', 'bety', 'dx', 'dy',
                                     'mux',
                                     'muy', 'env_x', 'env_y', 'aper_1', 'aper_2'])
            twiss = self.madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                                    BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                                    x=0.00, px=0.00, y=0.0, py=0.0, deltap=str(-0.000))
            _ = ax.add_patch(matplotlib.patches.Rectangle((twiss['s'][twiss['name'] == "plasma_merge:1"], -0.01), 10, 0.02, facecolor="peachpuff", edgecolor="sandybrown", lw=2))
            ax.set_xlim(750, twiss['s'][twiss['name'] == "plasma_merge:1"]+2)
            ax.set_ylim(-0.25, 0.05)
            for idx in range(np.size(twiss['l'])):
                if twiss['aper_1'][idx] > 0:
                    ts = ax.transData
                    coords  = ([(twiss['S'][idx] - twiss['L'][idx]/2), (total_chicane[idx] + total_chicane[idx - 1]) / 2])
                    if twiss['keyword'][idx] == "rbend":
                        tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], (
                                total_angle[idx]+total_angle[idx-1])/4)
                    else:
                        tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], (
                                    total_angle[idx] + total_angle[idx - 1]) / 2)
                    t = tr + ts
                    _ = ax.add_patch(
                        matplotlib.patches.Rectangle(
                            (twiss['S'][idx] - twiss['L'][idx], twiss["aper_1"][idx]/2 + (total_chicane[idx]+total_chicane[idx-1])/2), twiss['L'][idx], 0.3,
                            edgecolor='k', facecolor='lightgrey', transform=t))
                    _ = ax.add_patch(
                        matplotlib.patches.Rectangle(
                            (twiss['S'][idx] - twiss['L'][idx], -twiss["aper_1"][idx]/2 + (total_chicane[idx]+total_chicane[idx-1])/2 - 0.3), twiss['L'][idx], 0.3,
                            edgecolor='k', facecolor='lightgrey', transform=t))

            laser_size = (0.001 + (twiss['s'][twiss['name'] == "plasma_merge:1"] - twiss['s'][
                twiss['name'] == "laser.1:1"]) * np.sin(0.0005))

            for idx in range(np.size(twiss['l'])):
                if twiss['name'][idx] == 'laser.1:1':
                    _ = ax.add_patch(
                                matplotlib.patches.Rectangle(
                                    (twiss['s'][idx] - 0.05, -laser_size), 0.1, 2*laser_size,
                                    facecolor='m', edgecolor='m'))
            #     if twiss['aper_2'][idx] > 0:
            #         _ = ax.add_patch(
            #             matplotlib.patches.Rectangle(
            #                 (twiss['S'][idx] - twiss['L'][idx], 0.2), twiss['L'][idx], -0.2 + twiss["aper_2"][idx],
            #                 edgecolor='r', facecolor='w'))
            #         _ = ax.add_patch(
            #             matplotlib.patches.Rectangle(
            #                 (twiss['S'][idx] - twiss['L'][idx], -0.2), twiss['L'][idx], 0.2 - twiss["aper_2"][idx],
            #                 edgecolor='r', facecolor='w'))

            # ax.plot(survey['z'] + np.squeeze(np.multiply(1, xt, np.sin(theta))) - (0.001+(828.2955-survey['s'])*0.0005) * np.sin(theta),
            #         survey['x'] + np.squeeze(np.multiply(1, xt, np.cos(theta))) - (0.001 +(828.2955-survey['s'])*0.0005) *np.cos(theta), 'c-', linewidth=1.3)
            # ax.plot(survey['z'] - np.squeeze(np.multiply(1, xt, np.sin(theta))) + (0.001+(828.2955-survey['s'])*0.0005) *np.sin(theta),
            #         su  rvey['x'] - np.squeeze(np.multiply(1, xt, np.cos(theta))) + (0.001 +(828.2955-survey['s'])*0.0005) *np.cos(theta), 'c-', linewidth=1.3)

            # for idx in range(np.size(twiss['l'])):
            #
            #     if twiss['name'][idx] == 'laser.1:1':
            #         _ = ax1.add_patch(
            #             matplotlib.patches.Rectangle(
            #                 (twiss['s'][idx] - 0.025, -1), 0.05, 2,
            #                 facecolor='m', edgecolor='m'))

            #     if twiss['keyword'][idx] == 'quadrupole':
            #         _ = ax1.add_patch(
            #             matplotlib.patches.Rectangle(
            #                 (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], np.sign(twiss['k1l'][idx]),
            #                 facecolor='k', edgecolor='k'))
            #     elif twiss['keyword'][idx] == 'sextupole':
            #         _ = ax1.add_patch(
            #             matplotlib.patches.Rectangle(
            #                 (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], 0.9 * np.sign(twiss['k2l'][idx]),
            #                 facecolor='b', edgecolor='b'))
            #     elif twiss['keyword'][idx] == 'octupole':
            #         _ = ax1.add_patch(
            #             matplotlib.patches.Rectangle(
            #                 (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], 0.8 * np.sign(twiss['k3l'][idx]),
            #                 facecolor='r', edgecolor='r'))
            #     elif twiss['keyword'][idx] == 'rbend':
            #         _ = ax1.add_patch(
            #             matplotlib.patches.Rectangle(
            #                 (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
            #                 facecolor='g', edgecolor='g'))
            #     elif twiss['keyword'][idx] == 'vkicker':
            #         _ = ax1.add_patch(
            #             matplotlib.patches.Rectangle(
            #                 (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
            #                 facecolor='b', edgecolor='b'))
            #     elif twiss['keyword'][idx] == 'hkicker':
            #         _ = ax1.add_patch(
            #             matplotlib.patches.Rectangle(
            #                 (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
            #                 facecolor='b', edgecolor='b'))
            #     elif 'monitor' in twiss['keyword'][idx]:
            #         _ = ax1.add_patch(
            #             matplotlib.patches.Rectangle(
            #                 (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
            #                 facecolor='w', edgecolor='r'))
            #     elif twiss['keyword'][idx] == 'instrument':
            #         _ = ax1.add_patch(
            #             matplotlib.patches.Rectangle(
            #                 (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
            #                 facecolor='w', edgecolor='r'))

            # plt.gcf().subplots_adjust(bottom=0.15)
            # ax1.set(xlim=(750, twiss['s'][twiss['name'] == "plasma_merge:1"]), ylim=(-1, 1))
            # ax1.axes.get_yaxis().set_visible(False)
            # ax1.spines['top'].set_visible(False)
            # ax1.spines['right'].set_visible(False)
            # ax1.spines['bottom'].set_position('center')
            ax.grid(False)
            # ax.set_aspect('equal')
        plt.show()