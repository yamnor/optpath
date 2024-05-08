##
#  Path Optimization with String Method
##

require_relative 'qmengine'

##
#  Gaussian Class
##

class Gaussian < QMEngine
  attr_accessor :rootdir
  def initialize
    super
    @rootdir = "/usr/appli/g16/c01"
    @filename = {
      input:  "gau.inp",
      output: "gau.out",
      punch:  "gau.punch"
    }
  end
  def run(runflg)
    File.open(@nodelog, "w") do |fw|
      @node.each_with_index do |node, n|
        if runflg[n]
          scrdir = "#{@scratch}/#{node}"
          cmd = Array.new
          cmd << "mkdir -p #{scrdir}"
          cmd << "export g16root=#{@rootdir}"
          cmd << "export GAUSS_SCRDIR=#{scrdir}"
          cmd << ". #{@rootdir}/g16/bsd/g16.profile"
          cmd << "cd #{Dir.pwd}/#{node}"
          cmd << "g16 < #{@filename[:input]} >& #{@filename[:output]}"
          cmd << "if [ -e fort.7 ]"
          cmd << "  then mv fort.7 #{@filename[:punch]}"
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
      scfcyc = 0.0
      optcyc = 0
      filename = "#{node}/#{@filename[:output]}" 
      File.open(filename, "r") do |f|
        while line = f.gets
          if line =~ /SCF Done/
            energy  = (line.split)[4].to_f
            scfcyc += (f.gets.split)[1].to_f
            optcyc += 1
          end
        end
      end
      @record[n][:energy] = energy
      @record[n][:scfcyc] = scfcyc / optcyc
      @record[n][:optcyc] = optcyc
    end
    return @record
  end
  def read_energy
    @record = Array.new(@node.size).map{Hash.new}
    @node.each_with_index do |node, n|
      filename = "#{node}/#{@filename[:output]}" 
      File.open(filename, "r") do |f|
        while line = f.gets
          if line =~ /SCF Done/
            @record[n][:energy] = (line.split)[4].to_f
          end
        end
      end
    end
    return @record
  end
  def check_output(filename)
    conv = false
    term = false
    File.open(filename, "r") do |f|
      while line = f.gets
        if line =~ /SCF Done/
          conv = true
        end
        if line =~ /Normal termination of Gaussian/
          term = true
        end
      end
    end
    if conv == false
      raise RuntimeError.exception("Error: %s was not converged.\n"%[filename])
    end
    if term == false
      raise RuntimeError.exception("Error: %s was not terminated.\n"%[filename])
    end
  end
  def read_punch
    mol = Array.new(@node.size)
    @node.each_with_index do |node, n|
      filename = "#{node}/#{@filename[:punch]}"
      data = File.read(filename).split("\n").collect{|x| x.split}
      natoms = data.size / 2
      mol[n] = Molecule.new(natoms)
      for i in 0..natoms-1
        mol[n].elem[i] = NumberToSymbol[data[i][0].to_i]
        for j in 0..2
          mol[n].geom[3*i+j] = data[i][j+1].gsub(/D/,"e").to_f
          mol[n].grad[3*i+j] = data[i+natoms][j].gsub(/D/,"e").to_f
        end
      end
    end
    return mol
  end
end
