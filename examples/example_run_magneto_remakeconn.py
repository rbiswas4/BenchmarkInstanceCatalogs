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
def get_timestamp():
    ts = time.time()
    dt = datetime.datetime.fromtimestamp(ts)
    dts = dt.isoformat()
    print dts
    return dts

for i in range(5000):
    print ' Case number ', i 
    case.append(i)
    ts = get_timestamp()
    print type(ts)
    times.append(ts)
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
        print 'Numbers of queries', numQ
        shutil.rmtree(gcb.dirname)
        numQueries.append(numQ)
        success.append(1)
    except:
        numQueries.append(0)
        success.append(0)
    df = pd.DataFrame({'record': case, 'success': success,
                       'queries': numQueries, 'Time': times})
    df.to_csv('results.csv')
