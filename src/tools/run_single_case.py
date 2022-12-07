from PANDORA.Database import Database
from PANDORA.PMHC import PMHC
from PANDORA.Pandora import Pandora

db = Database.load()

ID = 'BA_32724'
peptide = 'GAMAKKGDEQKLRSA'
allele_type = ['HLA-DPA1*02:01', 'HLA-DPB1*01:01']
#outdir = '/projects/0/einf2380/data/pMHCII/3D_models/BA/80001_81000'
outdir = '/projects/0/einf2380/temp'

target = PMHC.Target(MHC_class = 'II',
        id = ID, peptide = peptide,
        allele_type = allele_type,
        use_netmhcpan=True)

mod = Pandora.Pandora(target, db, output_dir = outdir)
mod.model(clip_C_domain=True)