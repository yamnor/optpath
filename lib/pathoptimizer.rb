##
#  Path Optimization with String Method
##

require 'fileutils'

require_relative 'constants'
require_relative 'gaussian'
require_relative 'qchem'
require_relative 'molecule'
require_relative 'path'

class PathOptimizer
  def initialize(arg)
    ##
    #  Setting of Path Optimization
    ##
    @maxstep  = arg[:maxstep]
    @stepsize = arg[:stepsize]
    @nodedir  = arg[:node_dir]
    @pathdir  = arg[:path_dir]
    @pathlog  = arg[:path_log]
    #
    FileUtils.mkdir_p(@nodedir) unless FileTest.exist?(@nodedir)
    FileUtils.mkdir_p(@pathdir) unless FileTest.exist?(@pathdir)
    ##
    #  Path in Cartesian Coordinate
    ##
    @path = Path.new
    #
    mol = read_molecule(arg[:path_xyz])
    @path.make_from_geom(mol)
    #
    if arg[:nnodes] == @path.nnodes
      @path.normalize
    else
      @path.resize(arg[:nnodes])
    end
    ##
    #  Fixed Ends
    ##
    @path.optflg[ 0] = false
    @path.optflg[-1] = false
    #
    @nnodes = @path.nnodes
    ##
    #  Setting of QM Engine
    ##
    case arg[:qm_engine]
    when "gaussian"
      @qm = Gaussian.new
      @qm.engine = "gaussian"
    when "qchem"
      @qm = QChem.new
      @qm.engine = "qchem"
    end
    @qm.rootdir = arg[:qm_rootdir]
    @qm.node    = [*0..@nnodes-1].map{|n| "#{@nodedir}/%02d"%[n]}
    @qm.node.each{|dir| FileUtils.mkdir_p(dir) unless FileTest.exist?(dir)}
    @qm.nodelog = arg[:node_log]
    @qm.scratch = arg[:qm_scratch]
    @qm.reptag  = arg[:qm_reptag]
    @qm.ncpus   = arg[:qm_ncpus]
    #
    @qm.filename[:input]  = arg[:qm_input]
    @qm.filename[:output] = arg[:qm_output]
    @qm.filename[:punch]  = arg[:qm_punch]
    #
    @qm_record = Array.new(@nnodes).map{Hash.new}
    @qm_mol    = Array.new(@nnodes).map{Molecule.new}
    ##
    #  Template Files of QM Calculations
    ##
    @qm_sng  = arg[:qm_sng]
    @qm_opt  = arg[:qm_opt]
    @qm_grad = arg[:qm_grad]
    ##
    #  Temporary Data
    ##
    @memory = Array.new(@nnodes).map{Hash.new}
    #
  end
  def mkinp
    ##
    #  Make Input for Single Point Calculation
    ##
    @qm.make_inp(@qm_sng, @path)
    ##
    #  Output
    ##
    print_molecule("#{@pathdir}/sng.xyz", @path.mol)
  end
  def sng
    ##
    #  Single Point Calculation
    ##
    @qm.make_inp(@qm_sng, @path)
    runflg = Array.new(@nnodes, true)
    @qm.run(runflg)
    ##
    #  Output
    ##
    print_molecule("#{@pathdir}/sng.xyz", @path.mol)
  end
  def run
    ##
    #  Path Optimization
    ##
    for n in 0..@maxstep
      ##
      #  Force Calculation
      ##
      @qm.make_inp(@qm_grad, @path)
      @qm.run(@path.optflg)
      @qm_record = @qm.read_energy
      memcpy(@qm_record, datatype = :energy)
      @qm_mol = @qm.read_punch
      ##
      #  Path Optimization
      ##
      @path.evolve(@stepsize, @qm_mol)
      @path.normalize
      ##
      #  Output
      ##
      print_log("#{@pathlog}", n)
      print_dat("#{@pathdir}/%03d.dat"%[n])
      print_molecule("#{@pathdir}/%03d.xyz"%[n], @qm_mol)
      print_molecule("#{@pathdir}/%03d.gra"%[n], @qm_mol, datatype: :grad, unitconv: false)
    end
  end
  def memcpy(src, datatype)
    @nnodes.times do |n|
      @memory[n][datatype] = Marshal.load(Marshal.dump(src[n][datatype]))
    end
  end
  def read_molecule(filename, datatype: :geom, unitconv: true)
    mol = Array.new
    File.open(filename, "r") do |fr|
      while !fr.eof?
        mol_new = Molecule.new
        mol_new.read(fr, datatype: datatype, unitconv: unitconv)
        mol << mol_new
      end
    end
    return mol
  end
  def print_molecule(filename, mol, datatype: :geom, unitconv: true)
    File.open(filename, "w") do |fw|
      @nnodes.times do |n|
        fw.print mol[n].print(datatype: datatype, unitconv: unitconv)
      end
    end
  end
  def print_energy(filename)
    File.open(filename, "w") do |fw|
      @nnodes.times do |n|
        fw.print "%20.10f"%[@qm_record[n][:energy]]
      end
      fw.puts
    end
  end
  def print_dat(filename)
    label  = "# "
    label += "%20s"%["Energy / au"]
    label += "%20s"%["Delta-E / au"]
    label += "%20s"%["Gradient / au"]
    label += "%20s"%["DE / kcal mol-1"]
    File.open(filename, "w") do |fw|
      fw.puts label
      @nnodes.times do |n|
        value  = "  "
        value += "%20.10f"%[@qm_record[n][:energy]]
        value += "%20.10f"%[@memory[n][:edif]]
        value += "%20.10f"%[@memory[n][:grad]]
        value += "%20.10f"%[(@qm_record[n][:energy] - @qm_record[0][:energy]) * HartreeToKCalMol]
        fw.puts value
      end
    end
  end
  def print_log(filename, step)
    label = "%5s"%["NStep"]
    value = "%5d"%[step]
    ##
    #  Delta-E
    ##
    edif = Array.new(@nnodes, 0.0)
    @nnodes.times do |n|
      edif[n] = @qm_record[n][:energy] - @memory[n][:energy]
      @memory[n][:edif] = edif[n]
    end
    edif_max = edif.max_by{|x| x.abs}.abs
    edif_ave = edif.inject(0.0){|sum, x| sum += x.abs} / @nnodes
    label += "%18s%18s"%["Ene_Diff_Max", "Ene_Diff_Ave"]
    value += "%18.10f%18.10f"%[edif_max, edif_ave]
    ##
    #  Gradient
    ##
    grad = Array.new(@nnodes, 0.0)
    @nnodes.times do |n|
      count = 0.0
      @path.ndim.times do |i|
        grad[n] += @path.mol[n].perp[i]**2
        count += 1.0
      end
      grad[n] = Math.sqrt(grad[n] / count)
      @memory[n][:grad] = grad[n]
    end
    flexnode_grad = Array.new
    @nnodes.times do |n|
      if @path.optflg[n]
        flexnode_grad << grad[n]
      end
    end
    grad_max = flexnode_grad.max
    grad_ave = flexnode_grad.inject(0.0){|sum, x| sum += x} / flexnode_grad.size
    label += "%18s%18s"%["RMS_Grad_Max", "RMS_Grad_Ave"]
    value += "%18.10f%18.10f"%[grad_max, grad_ave]
    ##
    #  Print logfile
    ##
    if step == 0
      if File.exist?(filename)
        system("mv #{filename} #{filename}.1")
      end
      File.open(filename, "w") do |f|
        f.puts label
      end
    end
    File.open(filename, "a") do |f|
      f.puts value
    end
  end
end
