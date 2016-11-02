# -*- coding: utf-8 -*-
import numpy as np

from ..DropSizeDistribution import DropSizeDistribution
from ..io import common


def read_parsivel_nc(filename):
    '''
    Takes a filename pointing to a parsivel netcdf file and returns
    a drop size distribution object.

    Usage:
    dsd = read_parsivel(filename)

    Returns:
    DropSizeDistrometer object

    '''
    import xarray as xr

    ds = xr.open_dataset(filename)
    rain_rate = ds['rain_intensity'].values
    time = ds['time'].values
    nd = ds['drop_size_distribution'].values
    vd = ds['drop_velocity_distribution'].values

    reader = ParsedParsivelReader(time, nd, vd, rain_rate)
    dsd = DropSizeDistribution(reader)
    ds.close()

    return dsd


class ParsedParsivelReader(object):

    """
    ParsedParsivelReader class takes parsed arrays of values as its arguments
    and returns a reader object.
    """

    def __init__(self, time, nd, vd=None,
                 rain_rate=None, Z=None, num_particles=None):
        if rain_rate is None:
            rain_rate = []
        self.rain_rate = np.array(rain_rate)

        if Z is None:
            Z = []
        self.Z = np.array(Z)

        if num_particles is None:
            num_particles = []
        self.num_particles = np.array(num_particles)

        if vd is None:
            vd = []
        self.vd = np.array(vd)

        self.nd = np.power(10, np.array(nd))
        if self.nd.shape[0] == 32:
            self.nd = self.nd.T
            self.vd = self.vd.T
        self.time = time
        self.ndt = []

        self.pcm = np.reshape(self.pcm_matrix, (32, 32))

        self._prep_data()

        self.bin_edges = np.hstack(
            (0, self.diameter['data'] + np.array(self.spread['data']) / 2))

        self.bin_edges = common.var_to_dict(
            'bin_edges',
            self.bin_edges,
            'mm', 'Bin Edges')

    def _prep_data(self):
        self.fields = {}

        self.fields['rain_rate'] = common.var_to_dict(
            'Rain rate', self.rain_rate, 'mm/h', 'Rain rate')
        self.fields['Nd'] = common.var_to_dict(
            'Nd', np.ma.masked_equal(self.nd, np.power(10, -9.999)), 'm^-3 mm^-1',
            'Liquid water particle concentration')
        self.fields['Nd']['data'].set_fill_value(0)
        self.fields['num_particles'] = common.var_to_dict(
            'Number of Particles', self.num_particles,
            '', 'Number of particles')
        self.fields['terminal_velocity'] = common.var_to_dict(
            'Terminal Fall Velocity', self.vd,
            'm/s', 'Terminal fall velocity for each bin')
        self.time = {'data': self.time, 'units': None,
            'title': 'Time', 'full_name': 'Native file time'}

    diameter = common.var_to_dict(
        'diameter',
        np.array(
            [0.06, 0.19, 0.32, 0.45, 0.58, 0.71, 0.84, 0.96, 1.09, 1.22, 1.42, 1.67,
             1.93, 2.19, 2.45, 2.83, 3.35, 3.86, 4.38, 4.89, 5.66,
             6.7, 7.72, 8.76, 9.78, 11.33, 13.39, 15.45, 17.51, 19.57, 22.15, 25.24]),
        'mm', 'Particle diameter of bins')

    spread = common.var_to_dict(
        'spread',
        [
        0.129, 0.129, 0.129, 0.129, 0.129, 0.129, 0.129, 0.129, 0.129, 0.129, 0.257,
        0.257, 0.257, 0.257, 0.257, 0.515, 0.515, 0.515, 0.515, 0.515, 1.030, 1.030,
        1.030, 1.030, 1.030, 2.060, 2.060, 2.060, 2.060, 2.060, 3.090, 3.090],
        'mm', 'Bin size spread of bins')

    velocity = common.var_to_dict(
        'velocity',
        np.array(
            [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.3, 1.5, 1.7, 1.9,
             2.2, 2.6, 3, 3.4, 3.8, 4.4, 5.2, 6.0, 6.8, 7.6, 8.8, 10.4, 12.0, 13.6, 15.2,
             17.6, 20.8]),
        'm s^-1', 'Terminal fall velocity for each bin')

    v_spread = [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .2, .2, .2, .2, .2, .4,
                .4, .4, .4, .4, .8, .8, .8, .8, .8, 1.6, 1.6, 1.6, 1.6, 1.6, 3.2, 3.2]

    pcm_matrix = (
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
