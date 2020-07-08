"""
GET BEAM SIZE
R. Ramjiawan
Dec 2019
Optimise the beam size at injection based on seven quad strengths and two/four sextupole strengths
"""

import numpy as np
import scipy as scp


class getBeamSize:
    def __init__(self, q, n_particles, madx, no_quad, no_sext, no_oct, no_dist):
        self.q = q
        self.no_quad = no_quad
        self.no_sext = no_sext
        self.no_oct = no_oct
        self.no_dist = no_dist
        self.n_particles = n_particles
        self.madx = madx

    def get_beam_size(self):
        error_flag = 0
        i = 0
        for j in range(self.no_quad):
            self.madx.input("quad" + str(j) + "=" + str(self.q[i]) + ";")
            print("q" + str(j) + "=" + str(self.q[i]) + "")
            i = i + 1
        for j in range(self.no_sext):
            self.madx.input("sext" + str(j) + "=" + str(self.q[i]) + ";")
            print("s" + str(j) + "=" + str(self.q[i]) + "")
            i = i + 1
        for j in range(self.no_oct):
            self.madx.input("oct" + str(j) + "=" + str(self.q[i]) + ";")
            print("o" + str(j) + "=" + str(self.q[i]) + "")
            i = i + 1
        for j in range(self.no_dist):
            self.madx.input("dist" + str(j) + "=" + str(self.q[i]) + ";")
            print("a" + str(j) + "=" + str(self.q[i]) + "")
            i = i + 1

        var = []
        f = open('Ellipse1e6.tfs', 'r')  # initialize empty array
        for line in f:
            var.append(
                line.strip().split())
        f.close()

        init_dist = np.array(var[self.n_particles:])
        init_dist = init_dist[0:self.n_particles, 0:6]
        init_dist = init_dist.astype(np.float)

        
            self.madx.use(sequence='TT43')
            self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
            twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
        
        
