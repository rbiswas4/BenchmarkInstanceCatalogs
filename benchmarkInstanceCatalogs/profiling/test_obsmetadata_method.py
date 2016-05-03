from lsst.sims.catalogs.measures.instance import InstanceCatalog
import lsst.sims.catUtils.baseCatalogModels as bcm
from lsst.sims.catUtils.baseCatalogModels.GalaxyModels import GalaxyTileObj
from lsst.sims.utils import ObservationMetaData

obsMetaData = ObservationMetaData(boundType='box',
                                  pointingRA=0.,
                                  pointingDec=0.,
                                  boundLength=2.)


class galCopy(InstanceCatalog):
    column_outputs = ['galtileid', 'raJ2000', 'decJ2000', 'redshift']
    override_formats = {'raJ2000': '%8e', 'decJ2000': '%8e', 'a_d': '%8e',
                        'b_d': '%8e', 'pa_disk': '%8e', 'mass_stellar': '%8e',
                        'absmag_r_total': '%8e'}

def writeCat(redshiftmax=0.4):
    constr = 'redshift < {0}'.format(redshiftmax)
    galaxyTiled = GalaxyTileObj()
    galaxyBase = galCopy(galaxyTiled, obs_metadata=obsMetaData,
                         constraint=constr)
    galaxyBase.write_catalog('gB_test.csv')

writeCat()
