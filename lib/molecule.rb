##
#  Path Optimization with String Method
##

require_relative 'constants'

##
#  Molecule Class
##
class Molecule
  attr_accessor :natoms, :ndim, :elem, :geom, :grad, :perp
  def initialize(natoms = 0)
    @natoms = natoms
    @ndim = @natoms * 3
    @elem = Array.new(@natoms, "")
    @geom = Array.new(@natoms * 3, 0.0)
    @grad = Array.new(@natoms * 3, 0.0)
    @perp = Array.new(@natoms * 3, 0.0)
  end
  def read(fr, datatype: :geom, unitconv: true)
    if unitconv
      factor = AngstromToBohr
    else
      factor = 1.0
    end
    natoms = fr.gets.to_i
    initialize(natoms)
    fr.gets
    @natoms.times do |i|
      data = fr.gets.split
      @elem[i] = data.shift
      data = data.map{|x| x.to_f * factor}
      case datatype
      when :geom
        3.times do |j|
          @geom[3*i+j] = data[j]
        end
      when :grad
        3.times do |j|
          @grad[3*i+j] = data[j]
        end
      when :both
        3.times do |j|
          @geom[3*i+j] = data[i]
          @grad[3*i+j] = data[i+3]
        end
      end
    end
  end
  def duplicate
    mol = Molecule.new(@natoms)
    @natoms.times do |i|
      mol.elem[i] = @elem[i]
      3.times do |j|
        mol.geom[3*i+j] = @geom[3*i+j]
        mol.grad[3*i+j] = @grad[3*i+j]
      end
    end
    return mol
  end
  def print(datatype: :geom, unitconv: true, shownatoms: true)
    txt = ""
    if unitconv
      factor = BohrToAngstrom
    else
      factor = 1.0
    end
    if shownatoms
      txt += "%d\n"%[@natoms]
      txt += "\n"
    end
    @natoms.times do |i|
      txt += "%2s"%[@elem[i]]
      case datatype
      when :geom
        3.times do |j|
          txt += "%20.10f"%[@geom[3*i+j] * factor]
        end        
      when :grad
        3.times do |j|
          txt += "%20.10f"%[@grad[3*i+j] * factor]
        end        
      when :both
        3.times do |j|
          txt += "%20.10f"%[@geom[3*i+j] * factor]
        end        
        3.times do |j|
          txt += "%20.10f"%[@grad[3*i+j] * factor]
        end
      end
      txt += "\n"
    end
    return txt
  end
end
