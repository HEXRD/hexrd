#$ -q chess_fast.q
#$ -S /bin/bash
#$ -cwd
#$ -N mpi-test
#$ -pe sge_pe_rr 224
#$ -l h="lnx301|lnx302|lnx303|lnx304"

# Need to set up kerberos keytab to use this
/usr/bin/kinit -k -t /home/$USER/etc/$USER-keytab $USER

echo "Hostname: $(hostname)"

# Load mpi
module load mpi

# Activate conda environment
source /nfs/chess/user/pavery/virtualenvs/conda/bin/activate
conda activate hexrd

time mpiexec -npersocket 1 python nf-HEDM_test.py --check check.pkl --chunk-size 100 --ncpus 28
