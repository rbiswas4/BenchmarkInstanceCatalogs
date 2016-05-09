
import numpy as np
from lsst.sims.catalogs.measures.instance import InstanceCatalog
import lsst.sims.catUtils.baseCatalogModels as bcm
from lsst.sims.catUtils.baseCatalogModels.GalaxyModels import GalaxyTileObj
from lsst.sims.utils import ObservationMetaData

NUM = 3

obsMetaData = ObservationMetaData(boundType='box',
                                  pointingRA=0.,
                                  pointingDec=0.,
                                  boundLength=2.)

def tileObsMetaData(obsMetaData, num):
    ra, dec, size = obsMetaData.pointingRA, obsMetaData.pointingDec, obsMetaData.boundLength
    blen = size / num
    pras = np.arange(ra - size, ra + size, 2.0 *blen) + blen
    pdecs = np.arange(dec - size, dec + size, 2.0 * blen) + blen
    
    coords = np.dstack(np.meshgrid(pras, pdecs)).reshape(len(pras) * len(pdecs), 2)
    obsmdata = [ObservationMetaData(boundType='box',
                                  pointingRA=coords[i, 0],
                                  pointingDec=coords[i, 1],
                                  boundLength=blen) for i in range(len(coords))]
    # return coords
    return obsmdata, coords


class galCopy(InstanceCatalog):
    column_outputs = ['galtileid', 'raJ2000', 'decJ2000', 'redshift']
    override_formats = {'raJ2000': '%8e', 'decJ2000': '%8e', 'a_d': '%8e',
                        'b_d': '%8e', 'pa_disk': '%8e', 'mass_stellar': '%8e',
                        'absmag_r_total': '%8e'}


def writeCat(obsMD, redshiftmax=0.4, fname='gB-test', chunkSize=None):
    constr = 'redshift < {0}'.format(redshiftmax)
    galaxyTiled = GalaxyTileObj()
    print(len(obsMD))
    filenames = []
    for i, obsMetaData in enumerate(obsMD):
        filename = fname  + '{}'.format(i) + '.csv'
        filenames.append(filename)
        galaxyBase = galCopy(galaxyTiled, obs_metadata=obsMetaData, constraint=constr)
        galaxyBase.write_catalog(filename, chunkSize)

    return filenames
tileObsMD, coords = tileObsMetaData(obsMetaData, NUM)
print('length of tileObsMD is ', len(tileObsMD), NUM)
writeCat(obsMD=tileObsMD)
