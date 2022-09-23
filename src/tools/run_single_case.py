from PANDORA.Database import Database
from PANDORA.PMHC import PMHC
from PANDORA.Pandora import Pandora

db = Database.load("/projects/0/einf2380/softwares/PANDORA/PANDORA_files/data/csv_pkl_files/20220708_complete_db.pkl")
PDB_path = a.db_path.split('/data/')[0] + '/data/PDBs'
db.repath(PDB_path, save=False)

ID = 'BA_80109'
peptide = 'RGQALLVNSSQPWEP'
allele_type = ['HLA-DRB1*01:01']
outdir = '/projects/0/einf2380/data/pMHCII/3D_models/BA/80001_81000'

target = PMHC.Target(MHC_class = 'II',
        id = ID, peptide = peptide,
        allele_type = allele_type,
        use_netmhcpan=True)

mod = Pandora.Pandora(target, db, output_dir = outdir)
mod.model(clip_C_domain=True)