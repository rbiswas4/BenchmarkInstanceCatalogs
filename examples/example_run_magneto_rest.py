#!/usr/bin/env python

# To benchmark an instance Catalog, first look at
# the instance of CatalogDBObject
import numpy as np

from benchmarkInstanceCatalogs import QueryBenchMarks
from lsst.sims.catalogs.generation.db import CatalogDBObject
import lsst.sims.catUtils.baseCatalogModels as bcm
from lsst.sims.catalogs.measures.instance import InstanceCatalog

galDB = CatalogDBObject.from_objid('galaxyTiled')

# Create a child of the InstanceCatalog Class
class galCopy(InstanceCatalog):
    column_outputs = ['id', 'raJ2000', 'decJ2000', 'redshift']
    override_formats = {'raJ2000': '%8e', 'decJ2000': '%8e'}

# Sizes to be used for querying
boundLens = np.arange(0.05, 1.8, 0.05)

# Instantiate a benchmark object
opsimDBHDF ='/astro/users/rbiswas/data/LSST/OpSimData/storage.h5'
gcb = QueryBenchMarks.fromOpSimDF(instanceCatChild=galCopy, dbObject=galDB,
                                  opSimHDF=opsimDBHDF, boundLens=boundLens,
                                  constraints='r_ab < 24.0',
                                  numSamps=3, name='magneto_test_rless24')
# Look at the size 
print gcb.coords.size
print gcb.boundLens.size

# Now run results
gcb.aggregateResults()
fig = gcb.plots
fig.savefig('magneto_test_rless24.pdf')
