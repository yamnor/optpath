#!/user1/scl9/yamnor/apl/ruby/bin/ruby

#PBS -q SMALL
#PBS -N ethylene
#PBS -l select=1:ncpus=12:mpiprocs=12:mem=48gb
#PBS -l walltime=12:00:00

##
#  Path Optimization with the String Method
##

arg = {
  #
  # Input Files
  #
  path_xyz: "path.xyz",
  #
  qm_sng:   "qm.sng",
  qm_grad:  "qm.grad",
  #
  # Setting of Path Optimization
  #
  stepsize:  1.0,
  maxstep:  30,
  nnodes:   12,
  node_dir: "node",
  node_log: "node.log.#{ENV['PBS_JOBID']}",
  path_dir: "path",
  path_log: "path.log.#{ENV['PBS_JOBID']}",
  #
  # Setting of QM Engine
  #
  qm_rootdir: "/usr/appli/qchem/610",
  qm_scratch: "/scratch/qchem/#{ENV['PBS_JOBID']}",
  #
  qm_engine:  "qchem",
  qm_reptag:  "__geom__", # ... replace the variables indicated by this tag
  qm_input:   "qm.inp",
  qm_output:  "qm.out",
  qm_punch:   "qm.dat",
  qm_ncpus:   1,
}

$LOAD_PATH << '/user1/scl9/yamnor/apl/optpath/lib'

require 'pathoptimizer'

if ENV['PBS_O_WORKDIR'] != nil
  Dir.chdir(ENV['PBS_O_WORKDIR'])
end

po = PathOptimizer.new(arg)
po.sng
po.run
