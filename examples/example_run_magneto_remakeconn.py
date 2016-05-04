#!/usr/bin/env python

# To benchmark an instance Catalog, first look at
# the instance of CatalogDBObject
import numpy as np
import shutil
import time 
import datetime
from dateutil.parser import parse
import pandas as pd
from benchmarkInstanceCatalogs import QueryBenchMarks
from lsst.sims.catalogs.generation.db import CatalogDBObject
import lsst.sims.catUtils.baseCatalogModels as bcm
from lsst.sims.catalogs.measures.instance import InstanceCatalog


saveafter = 50
numrows = 5000
# Create a child of the InstanceCatalog Class
class galCopy(InstanceCatalog):
    column_outputs = ['id', 'raJ2000', 'decJ2000', 'redshift']
    override_formats = {'raJ2000': '%8e', 'decJ2000': '%8e'}

# Sizes to be used for querying
# boundLens = np.arange(0.02, 0.1, 0.02)
boundLens = np.array([0.02]) # , 0.1, 0.02)

# Instantiate a benchmark object
opsimDBHDF ='/astro/users/rbiswas/data/LSST/OpSimData/storage.h5'
case = []
numQueries = []
success = []
times = []
timeSec = []
def get_timestamp():
    ts = time.time()
    dt = datetime.datetime.fromtimestamp(ts)
    dts = dt.isoformat()
    return ts, dts

for i in range(numrows):
    print ' Case number ', i 
    case.append(i)
    ts, dts = get_timestamp()
    print type(ts)
    times.append(dts)
    timeSec.append(ts)
    try:
        galDB = CatalogDBObject.from_objid('galaxyTiled')
        print 'dbobject created '
        gcb = QueryBenchMarks.fromOpSimDF(instanceCatChild=galCopy,
                                          dbObject=galDB, opSimHDF=opsimDBHDF,
                                          boundLens=boundLens, numSamps=1,
                                          name='magneto_small_cobble')
        gcb.aggregateResults()
        # fig = gcb.plots
        numQ = len(gcb.df)
        shutil.rmtree(gcb.dirname)
        done = 1
    except:
        numQ = 0
        done = 0
    success.append(done)
    numQueries.append(numQ)
    print 'Numbers of queries', numQ, 'success', done
    if (i % saveafter) == 0:
        df = pd.DataFrame({'record': case, 'success': success,
            'queries': numQueries, 'Time': times,
            'timesec':timeSec}, index=case)
        df.to_csv('results_after.csv', index=False)
