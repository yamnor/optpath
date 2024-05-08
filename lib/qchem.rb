##
#  Path Optimization with String Method
##

require_relative 'qmengine'

##
#  QChem Class
##

class QChem < QMEngine
  attr_accessor :rootdir
  def initialize
    super
    @rootdir = "/usr/appli/qchem/542"
    @filename = {
      input:  "qm.inp",
      output: "qm.out",
      punch_geom: "qm.geom",
      punch_grad: "qm.grad"
    }
  end
  def run(runflg)
    File.open(@nodelog, "w") do |fw|
      @node.each_with_index do |node, n|
        if runflg[n]
          scrdir = "#{@scratch}"
          scrnum = "%02d"%[n]
          cmd = Array.new
          cmd << "mkdir -p #{scrdir}"
          cmd << "export QC=#{@rootdir}"
          cmd << "export QCSCRATCH=#{scrdir}"
          cmd << ". #{@rootdir}/qcenv.sh"
          cmd << "cd #{Dir.pwd}/#{node}"
          cmd << "if [ -e #{@filename[:output]} ]"
          cmd << "  then mv #{@filename[:output]} #{@filename[:output]}.1"
          cmd << "fi"
          cmd << "qchem -save -nt #{@ncpus} #{@filename[:input]} #{@filename[:output]} #{scrnum}"
          cmd << "if [ -e #{scrdir}/#{scrnum}/molecule ]"
          cmd << "  then mv #{scrdir}/#{scrnum}/molecule #{@filename[:punch_geom]}"
          cmd << "fi"
          cmd << "if [ -e #{scrdir}/#{scrnum}/GRAD ]"
          cmd << "  then mv #{scrdir}/#{scrnum}/GRAD #{@filename[:punch_grad]}"
          cmd << "fi"
          cmd << "rm -r #{scrdir}"
          cmd = cmd.join("; ")
          fw.puts "Running in %s (command: %s)"%[node, cmd]
          system cmd
          fw.puts "Terminated in %s (status: %s)"%[node, $?]
        end
      end
    end
  end
  def read_output
    @record = Array.new(@node.size).map{Hash.new}
    @node.each_with_index do |node, n|
      energy = 0.0
      filename = "#{node}/#{@filename[:output]}" 
      File.open(filename, "r") do |f|
        while line = f.gets
         if line =~ /Total energy in the final/
            energy  = (line.split)[8].to_f
          end
        end
      end
      @record[n][:energy] = energy
    end
    return @record
  end
  def read_energy
    @record = Array.new(@node.size).map{Hash.new}
    @node.each_with_index do |node, n|
      filename = "#{node}/#{@filename[:punch_grad]}"
      @record[n][:energy] = (File.open(filename).readlines)[1].to_f
    end
    return @record
  end
  def read_punch
    mol = Array.new(@node.size)
    @node.each_with_index do |node, n|
      file_geom = "#{node}/#{@filename[:punch_geom]}"
      file_grad = "#{node}/#{@filename[:punch_grad]}"
      data_geom = File.open(file_geom).readlines.values_at(2..-2).map{|x| x.split}
      data_grad = File.open(file_grad).readlines.values_at(3..-2).map{|x| x.split}
      natom  = data_geom.size
      mol[n] = Molecule.new(natom)
      natom.times do |i|
        mol[n].elem[i] = data_geom[i].shift
        3.times do |j|
          mol[n].geom[3*i+j] = data_geom[i][j].to_f * AngstromToBohr
          mol[n].grad[3*i+j] = data_grad[i][j].to_f
        end
      end
    end
    return mol
  end
  def read_geom
    mol = Array.new(@node.size)
    @node.each_with_index do |node, n|
      file_geom = "#{node}/#{@filename[:punch_geom]}"
      data_geom = File.open(file_geom).readlines.values_at(2..-2).map{|x| x.split}
      natom  = data_geom.size
      mol[n] = Molecule.new(natom)
      natom.times do |i|
        mol[n].elem[i] = data_geom[i].shift
        3.times do |j|
          mol[n].geom[3*i+j] = data_geom[i][j].to_f * AngstromToBohr
        end
      end
    end
    return mol
  end
end
