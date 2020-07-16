from cpymad.madx import Madx
import numpy as np
import plot_save_output as plot
import matplotlib.pyplot as plt
import matplotlib

madx = Madx(stdout=False)
madx.call(file='general_tt43_python.madx')
madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
madx.use(sequence='TT40TT41')
madx.input("DeltaP = 0.00035;")
madx.input("emitta = 8.21e-9;")

madx.input(
    "sigma_x := 1000000 *(sqrt((table(twiss, betx)*emitta) + (abs(table(twiss,dx))*DeltaP)*(abs(table(twiss,dx))*DeltaP)));")
madx.input(
    "sigma_y :=  1000000*(sqrt((table(twiss, bety)*emitta) + (abs(table(twiss,dy))*DeltaP)*(abs(table(twiss,dy))*DeltaP)));")

madx.select(flag='twiss',
                 column=['name', 'keyword', 'sigma_x', 'sigma_y', 'k3l', 's', 'l', 'betx', 'bety', 'dx', 'dy', 'mux',
                         'muy', 'env_x', 'env_y', 'aper_1', 'aper_2'])
twiss = madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
                   BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
                   x=0.00, px=0.00, y=0.0, py=0.0, file="twiss.out")

# madx.input("DeltaP = 0.00035;")
# madx.input("emitta = 8.21e-9;")
# madx.input("orbitError = 0.002;")
# madx.input("alignmentError = 0.003;")
# madx.input(
#     "sigma_x := 6*(sqrt((table(twiss, betx)*emitta) + (abs(table(twiss,dx))*DeltaP)*(abs(table(twiss,dx))*DeltaP)));")
# madx.input(
#     "sigma_y :=  6*(sqrt((table(twiss, bety)*emitta) + (abs(table(twiss,dy))*DeltaP)*(abs(table(twiss,dy))*DeltaP)));")
# madx.select(flag='interpolate', step=0.05)
madx.survey(x0=-2314.71736,
            y0=2401.75556,
            z0=4456.10249,
            theta0=-2.17482033,
            phi0=0.000178,
            psi0=0,
            file="ptc_survey.out")

plot = plot.Plot(madx)
plot.errors()
# plot.survey()
# plot.errors()
#plot.laser_mirror()

# plot.diffTwiss()
# plot.plot_phase_advance()
# plot.twiss()

# plot.offset()
# plot.survey()
# madx.call(file='original_line.madx')
# madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
# madx.use(sequence='tt40tt41_original')
#
# twiss = madx.twiss(BETX=27.931086, ALFX=0.650549, DX=-0.557259, DPX=0.013567,
#                    BETY=120.056512, ALFY=-2.705071, DY=0.0, DPY=0.0,
#                    x=0.00, px=0.00, y=0.0, py=0.0, file="twiss_original.out")
# plot.twiss()
# madx.input("DeltaP = 0.00035;")
# madx.input("emitta = 8.21e-9;")
# madx.input("orbitError = 0.002;")
# madx.input("alignmentError = 0.003;")
# madx.input(
#     "sigma_x := 6*(sqrt((table(twiss, betx)*emitta) + (abs(table(twiss,dx))*DeltaP)*(abs(table(twiss,dx))*DeltaP)));")
# madx.input(
#     "sigma_y :=  6*(sqrt((table(twiss, bety)*emitta) + (abs(table(twiss,dy))*DeltaP)*(abs(table(twiss,dy))*DeltaP)));")
# # madx.select(flag='interpolate', step=0.05)
# madx.survey(x0=-2314.71736,
#             y0=2401.75556,
#             z0=4456.10249,
#             theta0=-2.17482033,
#             phi0=0.000178,
#             psi0=0,
#             file="ptc_survey_original.out")
#
