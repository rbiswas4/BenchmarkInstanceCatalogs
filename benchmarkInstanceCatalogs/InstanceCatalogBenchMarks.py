#!/usr/bin/env python

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from lsst.sims.utils import ObservationMetaData

__all__ = ['QueryBenchMarks']


class QueryBenchMarks(object):

    """
    Class to benchmark `lsst.sims.catalogs.measures.instance.InstanceCatalog` 
    queries to fatboy for different objects using different instances of 
    `lsst.sims.utils.ObservationMetaData`. 

    Important features are the ability to conveniently
    - time queries over regions of varying sizes at different locations in 
    the sky input through a list of sizes, and field centers in Ra, Dec in 
    degrees requested.
    - Instantiate the set of pointings as unique pointings from by choosing 
    LSST fields 
    - Serialize the output of past queries and record set of outstanding
    requests to checkpoint the state
    - Restart from checkpoints 
    - record information on both the database being used, internet and user
    information
    - record time stamps
    - record data quantity
    - Do meaningful calculations / plot results even when the number of samples
    is 1.
     """

    def __init__(self, instanceCatChild, dbObject, boundLens, Ra, Dec,
                 name='catsim', numSamps=3, mjd=572013.,
                 constraints=None, checkpoint=True, df=None):
        """
        instanceCatChild : instance of a class inheriting from
            `lsst.sims.catalogs.measures.instances.InstanceCatalog`, mandatory
            Class that will be queried on the database
        boundLens : array like, of floats
                size of `lsst.sims.utils.ObsMetaData` BoundLengths in degrees
        Ra : arrayLike, floats, degrees 
            length should be sum(num_samples)
        Dec : arrayLike, floats, degrees
            length should be sum(num_samples)
        numSamps : arrray-like or float. If array like must have the same len
            as boundLens

        """
        self.checkpoint = checkpoint
        self.constraints = constraints
        self.name = name
        self.numSamps = numSamps
        self.mjd = mjd
        self.dbObject = dbObject
        self.instanceCatChild = instanceCatChild

        if len(Ra) != len(Dec):
            raise ValueError(
                'the lengths of the ra and dec array have to be the same')

        # setup the radii of circles we want to query
        # Each radius may be sampled a number of times indicated by numSamps
        boundLens = np.asarray(boundLens)
        boundLens = boundLens.repeat(self.numSamps)
        self.boundLens = boundLens

        # Put if clause if numsamps is not array
        self.numSamps = np.broadcast_arrays(numSamps, self.boundLens)[0]

        self.Ra = Ra
        self.Dec = Dec
        self.coords = np.asarray(zip(self.Ra, self.Dec))

        # The order is shuffled to prevent all samples of a particular radius
        # being evaluated at the same time
        np.random.shuffle(boundLens)
        np.random.shuffle(self.coords)

        if df is None:
            np.savetxt(name + '_Init_boundlens.dat', self.boundLens)
            np.savetxt(name + '_Init_numSamps.dat', self.numSamps)
            np.savetxt(name + '_Init_coords.dat', self.coords)
            with open(name + 'constraints.dat','w') as fp:
                if self.constraints is None:
                    fp.write('None')
                else:
                    fp.write(self.constraints)
        self.numRequests = len(self.boundLens)

        self.df = df

    @property
    def boundLength_fname(self):
        return self.name + '_boundLength.dat'

    @property
    def boundLength_fname(self):
        return self.name + '_coords.dat'

    @classmethod
    def fromCheckPoint(cls, instanceCatChild, dbObject, cacheDir, name,
                       dffname=None,
                       mjd=572013., constraints=None, checkpoint=True):
        """
        Instantiate class from saved checkpoint

        Parameters
        ----------
        cachedir

        """
        import os
        import pandas as pd

        if dffname is None:
            dffname = name + 'constraints.dat'
        df = pd.read_hdf(dffname, 'table')
        boundLengthfname = os.path.join(cacheDir, name + '_boundLens.dat')
        boundLens = np.loadtxt(boundLengthfname).flatten()
        coordsfname = os.path.join(cacheDir, name + '_coords.dat')
        coords = np.loadtxt(coordsfname)
        ra, dec = zip(*coords)
        Ra = np.asarray(ra)
        Dec = np.asarray(dec)

        numSamps = np.loadtxt(name + '_numSamps.dat')
        return cls(instanceCatChild=instanceCatChild, dbObject=dbObject,
                   boundLens=boundLens, Ra=Ra, Dec=Dec, name=name,
                   numSamps=numSamps, mjd=mjd, constraints=constraints,
                   checkpoint=checkpoint, df=df)

    @classmethod
    def fromOpSimDF(cls, instanceCatChild, dbObject, opSimHDF, boundLens,
                    mjd=57210, numSamps=1,
                    constraints=None, checkpoint=True, name='test',
                    summaryTable='table', df=None):
        """
        Instantiate using different LSST fields of view from an OpSim run

        boundLens : array-like, mandatory, degrees
            array of boundLength values for obsMetaData 
        mjd :
        numSamps :
        constraints :
        checkpoints :
        opSimHDF :
        summaryTable :
        """
        boundLens = np.asarray(boundLens)
        numLengths = len(boundLens)

        opsimDf = pd.read_hdf(opSimHDF, 'table')
        fieldIds = opsimDf.fieldID.unique()
        fid = np.random.choice(fieldIds, size=numLengths * numSamps,
                               replace=False)
        x = opsimDf[opsimDf['fieldID'].isin(fid)].groupby('fieldID')
        k = x.groups.keys()
        coords = map(lambda y: x.get_group(y)[['fieldRA', 'fieldDec']].iloc[
                     0].apply(np.degrees).as_matrix(), k)
        ra, dec = zip(*coords)
        ra = np.asarray(ra) - 180.0
        dec = np.asarray(dec)

        return cls(instanceCatChild=instanceCatChild, dbObject=dbObject,
                   boundLens=boundLens, Ra=ra, Dec=dec, numSamps=numSamps,
                   mjd=mjd, name=name, constraints=constraints, df=df)

    @property
    def results(self):

        if self.df is not None:
            self._results = QueryBenchMarks.benchmarkResults(self.df)
        return self._results

    @staticmethod
    def benchmarkResults(df):
        grouped = df.groupby('boundLen')
        boundLens = grouped.groups.keys()

        mydict = dict()
        mydict['boundLen'] = np.array(boundLens)
        mydict['coords'] = map(lambda x: zip(grouped.get_group(x)['Ra'].values,
                                             grouped.get_group(x)['Dec'].values), boundLens)
        mydict['mjd'] = map(
            lambda x: grouped.get_group(x).Mjd.values, boundLens)
        mydict['deltaTimeList'] = map(
            lambda x: grouped.get_group(x).deltaT.values, boundLens)
        mydict['numObjectList'] = map(
            lambda x: grouped.get_group(x).numObjects.values, boundLens)
        mydict['numObjects'] = map(
            lambda x: grouped.get_group(x).numObjects.mean(), boundLens)
        mydict['numObjectsWidth'] = map(
            lambda x: grouped.get_group(x).numObjects.std(), boundLens)
        mydict['deltaTimeFullList'] = map(
            lambda x: grouped.get_group(x).deltaTFull.values, boundLens)
        mydict['deltaTime'] = map(
            lambda x: grouped.get_group(x).deltaT.mean(), boundLens)
        mydict['deltaTimeFull'] = map(
            lambda x: grouped.get_group(x).deltaTFull.mean(), boundLens)
        mydict['deltaTwidth'] = map(
            lambda x: grouped.get_group(x).deltaT.std(), boundLens)
        mydict['deltaTFullwidth'] = map(
            lambda x: grouped.get_group(x).deltaTFull.std(), boundLens)
        results = pd.DataFrame(mydict)
        results.sort('boundLen', inplace=True)
        return results

    def benchMarkLen(self, ind=0, unique=True):
        """
        Benchmark the query with boundLen value at self.boundLen[ind]

        Parameters
        ----------
        ind : integer, optional, defaults to 0

        """
        results = []
        results.append(self.queryResult(self.boundLens[ind],
                                        coords=self.coords[ind],
                                        Mjd=self.mjd))

        # Remove the evaluated benchmark from the list of requested
        # benchmarks
        self.coords = np.delete(self.coords, [ind], axis=0)
        self.boundLens = np.delete(self.boundLens, [ind])
        self.numSamps = np.delete(self.numSamps, [ind])

        df = pd.DataFrame(results, 
                          columns=['boundLen', 'Ra', 'Dec', 'Mjd', 'numObjects',
                                   'deltaT', 'deltaTFull'])
        return df

    def serialize(self, savedtype='hdf'):
        """
        record the state of the benchmark by writing the inputs to the remaining
        benchmark requests to files, and saving the all the completed benchmarks
        to a file
        """
        if savedtype == 'hdf':
            self.df.to_hdf(self.name + '.hdf', 'table')
        else:
            raise ValueError(
                    'Saving the results to savetype file not implemented')
        # Save boundLens, coords, numSamps and MJD
        np.savetxt(self.name + '_boundLens.dat', self.boundLens)
        np.savetxt(self.name + '_coords.dat', self.coords)
        np.savetxt(self.name + '_numSamps.dat', self.numSamps)


    def aggregateResults(self):

        for i, boundLen in enumerate(self.boundLens):
            print 'boundLen used', i, boundLen
            df = self.benchMarkLen()
            if self.df is None:
                self.df = df
            else:
                self.df = pd.concat([self.df, df], ignore_index=True)
            self.serialize()

    @property
    def plots(self):
        """
        figure object having plots of the results 
        """

        fig = self.plotBenchMarks(results=self.results)
        fig.savefig(self.name + '.pdf')
        return fig

    @staticmethod
    def plotBenchMarks(results, overplotonfig=None, **kwargs):
        """

        Parameters
        ----------
        results: `pandas.DataFrame` with certain columns 
        """

        res = results
        res.sort('boundLen', inplace=True)
        if overplotonfig is None:
            fig = plt.figure()
            # fig, ax = plt.subplots(2, 2, sharex=[0,2], sharey=True)
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2, sharey=ax1)
            ax3 = fig.add_subplot(2, 2, 3, sharex=ax1)
            ax4 = fig.add_subplot(2, 2, 4, sharex=ax2, sharey=ax3)
        else:
            fig = overplotonfig
        
        # Check if the format of points is specified through kwargs, 
        # Else put in default values
        if 'fmto' in kwargs.keys():
            myfmto = kwargs['fmto']
        else:
            myfmto = 'ko'
        if 'fmts' in kwargs.keys():
            myfmts = kwargs['fmts']
        else:
            myfmts = 'rs'


        # Plot the statistics of the query times with estimates of uncertainty
        fig.axes[0].errorbar(res.boundLen, res.deltaTime, res.deltaTwidth,
        # ax[0, 0].errorbar(res.boundLen, res.deltaTime, res.deltaTwidth,
                          fmt=myfmto)
        fig.axes[1].errorbar(np.log10(res.numObjects), res.deltaTime,
        # ax[0, 1].errorbar(np.log10(res.numObjects), res.deltaTime,
                          xerr=np.log(10) * res.numObjectsWidth /
                          res.numObjects,
                          yerr=res.deltaTwidth, fmt=myfmto)
        # yt = fig.axes[1].get_yticklabels()
        # print yt
        # fig.axes[1].set_yticklabels(yt, visible=False)
        fig.axes[2].errorbar(res.boundLen, res.deltaTimeFull, res.deltaTFullwidth,
        # ax[1, 0].errorbar(res.boundLen, res.deltaTimeFull, res.deltaTFullwidth,
                          fmt=myfmto)
        fig.axes[3].errorbar(np.log10(res.numObjects), res.deltaTimeFull,
        # ax[1, 1].errorbar(np.log10(res.numObjects), res.deltaTimeFull,
                          xerr=np.log(10) * res.numObjectsWidth /
                          res.numObjects,
                          yerr=res.deltaTFullwidth, fmt=myfmto)

        # yt = fig.axes[3].get_yticklabels()
        # print yt
        # fig.axes[3].set_yticklabels(yt, visible=False)
        # Plot simple, proportional to area query times to guide the eye
        fig.axes[0].plot(res.boundLen,
        # ax[0, 0].plot(res.boundLen,
                      (res.deltaTime.iloc[0] / res.boundLen.iloc[0] ** 2)
                      * res.boundLen ** 2.0, myfmts)
        fig.axes[1].plot(np.log10(res.numObjects),
        # ax[0, 1].plot(np.log10(res.numObjects),
                      (res.deltaTime.iloc[0] / res.boundLen.iloc[0] ** 2)
                      * res.boundLen ** 2.0, myfmts)

        # Set up axes labels  and grids
        # ax[0, 0].set_ylabel('Query Time')
        fig.axes[0].set_ylabel('Query Time')
        fig.axes[2].set_ylabel('Query Time for Focal Plane')
        # ax[1, 0].set_ylabel('Query Time for Focal Plane')
        fig.axes[2].set_xlabel('Circle Radius')
        # ax[1, 0].set_xlabel('Circle Radius')
        fig.axes[3].set_xlabel(r'$\log_{10}(num Objects)$')
        # ax[1, 1].set_xlabel(r'$\log_{10}(num Objects)$')

        # put in a grid
        map(lambda x: x.grid(True), fig.axes)

        # tighten layout of plots
        fig.set_tight_layout(True)

        return fig

    def queryResult(self, boundLen, coords, Mjd, fieldRadius=1.75):
        """
        benchmarks a single query to download an instance catalog
        of given boundLen, and pointing defined by Ra, Dec in degrees.

        Parameters
        ----------
        boundLen : float, degrees
        coords : array like of size 2, mandatory, degrees
            iterable of Ra and Dec in degrees
        # Ra : float, degrees
        # Dec : float, degrees
        fieldRadius : Radius of field of view in degrees
        """
        Ra = coords[0]
        Dec = coords[1]
        myObsMD = ObservationMetaData(boundType='circle',
                                      boundLength=boundLen,
                                      unrefractedRA=Ra,
                                      unrefractedDec=Dec,
                                      site=None,
                                      bandpassName=[
                                          'u', 'g', 'r', 'i', 'z', 'y'],
                                      mjd=Mjd)
        tstart = time.time()
        icc = self.instanceCatChild(db_obj=self.dbObject, obs_metadata=myObsMD,
                                    constraint=self.constraints)
        icc.write_catalog("icc_tmp.dat")
        tend = time.time()
        deltaT = tend - tstart
        numObjects = sum(1 for _ in open('icc_tmp.dat'))

        return [boundLen, Ra, Dec, Mjd, numObjects, deltaT, deltaT * (np.float(fieldRadius) / np.float(boundLen)) ** 2.0]

if __name__ == '__main__':

    # To benchmark an instance Catalog, first look at
    # the instance of CatalogDBObject
    galDB = CatalogDBObject.from_objid('galaxyTiled')

    # Create a child of the InstanceCatalog Class
    class galCopy(InstanceCatalog):
        column_outputs = ['id', 'raJ2000', 'decJ2000', 'redshift']
        override_formats = {'raJ2000': '%8e', 'decJ2000': '%8e'}

    # Sizes to be used for querying
    boundLens = np.arange(0.01, 0.2, 0.03)

    # Instantiate a benchmark object
    opsimDBHDF = '/Users/rbiswas/data/LSST/OpSimData/storage.h5'
    gcb = QueryBenchMarks.fromOpSimDF(instanceCatChild=galCopy, dbObject=galDB,
                                      opSimHDF=opsimDBHDF, boundLens=boundLens,
                                      numSamps=3, name='test')
    # Look at the size
    print gcb.coords.size
    print gcb.boundLens.size

    # Now run results
    gcb.aggregateResults()
    # Obtain plots
    fig = gcb.plots
    fig.savefig('catsim_test.pdf')
