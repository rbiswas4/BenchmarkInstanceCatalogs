from __future__ import division, print_function, absolute_import
import os
import pymssql
from lsst.sims.catalogs.measures.instance import InstanceCatalog
import lsst.sims.catUtils.baseCatalogModels as bcm
from lsst.sims.catalogs.generation.db import CatalogDBObject
from lsst.sims.catUtils.baseCatalogModels.GalaxyModels import GalaxyTileObj
from lsst.sims.utils import ObservationMetaData
from lsst.utils import getPackageDir
import time

from lsst.daf.persistence import DbAuth
import lsst.pex.config as pexConfig
import __builtin__

try:
    __builtin__.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func
    __builtin__.profile = profile



@profile
def query_time():

    config = bcm.BaseCatalogConfig()
    config.load(os.path.join(getPackageDir("sims_catUtils"), "config", "db.py"))
    DBConnection = pymssql.connect(user=DbAuth.username(config.host, config.port),
                               password=DbAuth.password(config.host, config.port),
                               database=config.database, port=config.port)
    db = DBConnection.cursor()

    query = """SELECT id, ra, dec, redshift FROM galaxy WHERE redshift < 0.4 AND ra < 2 and ra > -2 and dec < 2 and dec > -2"""
    db.execute(query)
    x = db.fetchall()
    #print(len(x), len(x[0]))
    return len(x)
tstart = time.time()
query_time()
tend = time.time()
print(tend - tstart)
