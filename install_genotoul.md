devel/python/Python-3.12.4
python -m venv physigym
source physigym/bin/activate
cd PhysiGym/ # I should be into PhysiGym
python install_physigym.py tumor_immune_base
cd ../PhysiCell/
make load=physigym_tumor_immune_base




# devel/Miniconda/Miniconda3