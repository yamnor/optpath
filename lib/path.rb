##
#  Path Optimization with String Method
##

require 'spliner'

require_relative 'molecule'

##
#  Path Class
##
class Path
  attr_accessor :nnodes,
                :ndim,
                :mol,
                :optflg
  def initialize(nnodes = 0)
    @nnodes = nnodes
    @ndim   = 0
    @mol    = Array.new(@nnodes).map{Molecule.new}
    @optflg = Array.new(@nnodes, true)
  end
  def evolve(stepsize, mol)
    @nnodes.times do |n|
      ##
      #  Set projector
      ##
      proj = projector
      ##
      #  Evolve the path
      ##
      if @optflg[n]
        @ndim.times do |i|
          @mol[n].geom[i] = @mol[n].geom[i] - mol[n].grad[i] * stepsize
        end
      end
      ##
      #  Calculate gradient perpendicular to the path
      ##
      @ndim.times do |i|
        grad_perp = 0.0
        @ndim.times do |j|
          grad_perp += proj[n][i][j] * mol[n].grad[j]
        end
        @mol[n].perp[i] = grad_perp
      end
    end
  end
  def normalize
    x = Array.new(@nnodes, 0.0)
    for n in 1..@nnodes-1
      dist = 0.0
      @ndim.times do |i|
        dist += (@mol[n].geom[i] - @mol[n-1].geom[i])**2
      end
      dist = Math.sqrt(dist)
      x[n] = x[n-1] + dist
    end
    for n in 0..@nnodes-1
      x[n] = x[n] / x[@nnodes-1]
    end
    @ndim.times do |i|
      y = Array.new(@nnodes, 0.0)
      for n in 0..@nnodes-1
        y[n] = @mol[n].geom[i]
      end
      spl = Spliner::Spliner.new(x, y)
      y_new = spl[(0.0..1.0).step(1.0/(@nnodes-1))]
      for n in 0..@nnodes-1
        @mol[n].geom[i] = y_new[n]
      end
    end
  end
  def resize(nnodes_new)
    x = Array.new(@nnodes, 0.0)
    for n in 1..@nnodes-1
      dist = 0.0
      @ndim.times do |i|
        dist += (@mol[n].geom[i] - @mol[n-1].geom[i])**2
      end
      dist = Math.sqrt(dist)
      x[n] = x[n-1] + dist
    end
    @nnodes.times do |n|
      x[n] = x[n] / x[@nnodes-1]
    end
    mol_old = Array.new(@nnodes).map{Molecule.new}
    @nnodes.times do |n|
      mol_old[n] = @mol[n].duplicate
    end
    @mol = Array.new(nnodes_new).map{Molecule.new}
    ##
    #  Copy the basic information (@natoms, @elem) of molecule
    ##
    nnodes_new.times do |n|
      @mol[n] = mol_old.first.duplicate
    end
    ##
    # 
    ##
    @ndim.times do |i|
      y = Array.new(@nnodes, 0.0)
      @nnodes.times do |n|
        y[n] = mol_old[n].geom[i]
      end
      spl = Spliner::Spliner.new(x, y)
      y_new = spl[(0.0..1.0).step(1.0/(nnodes_new-1))]
      nnodes_new.times do |n|
        @mol[n].geom[i] = y_new[n]
      end
    end
    @nnodes = nnodes_new
    @optflg = Array.new(@nnodes, true)
  end
  def make_from_geom(mol)
    @nnodes = mol.size
    @ndim   = mol.first.ndim
    @mol = Array.new(@nnodes).map{Molecule.new}
    @nnodes.times do |n|
      @mol[n] = mol[n].duplicate
    end
    @optflg = Array.new(@nnodes, true)
  end
  def check_nnodes(obj)
    if obj.size != @nnodes
      raise "Number of nodes does not match with %s"%[obj.to_s]
    end
  end
  ##
  #  Bisection Tangent [Henkelman]
  ##
  def calc_dzda
    dzda = Array.new(@nnodes).map{Array.new(@ndim)}
    @ndim.times do |i|
      dzda[ 0][i] = @mol[ 1].geom[i] - @mol[ 0].geom[i]
      dzda[-1][i] = @mol[-1].geom[i] - @mol[-2].geom[i]
    end
    dzda_fw = Array.new(@nnodes).map{Array.new(@ndim)}
    dzda_bw = Array.new(@nnodes).map{Array.new(@ndim)}
    for n in 1..@nnodes-2
      @ndim.times do |i|
        dzda_fw[n][i] = @mol[n+1].geom[i] - @mol[n].geom[i]
        dzda_bw[n][i] = @mol[n].geom[i] - @mol[n-1].geom[i]
      end
      norm_fw = 0.0
      norm_bw = 0.0
      @ndim.times do |i|
        norm_fw += dzda_fw[n][i]**2
        norm_bw += dzda_bw[n][i]**2
      end
      norm_fw = Math.sqrt(norm_fw)
      norm_bw = Math.sqrt(norm_bw)
      @ndim.times do |i|
        dzda_fw[n][i] /= norm_fw
        dzda_bw[n][i] /= norm_bw
        dzda[n][i] = dzda_fw[n][i] + dzda_bw[n][i]
      end
    end
    @nnodes.times do |n|
      norm = 0.0
      @ndim.times do |i|
        norm += dzda[n][i]**2
      end
      norm = Math.sqrt(norm)
      @ndim.times do |i|
        dzda[n][i] /= norm
      end
    end
    return dzda
  end
  def projector
    dzda = calc_dzda
    proj = Array.new(@nnodes).map{Array.new(@ndim)}
    @nnodes.times do |n|
      @ndim.times do |i|
        proj[n][i] = Array.new(@ndim)
        @ndim.times do |j|
          if i == j
            proj[n][i][j] = 1.0 - dzda[n][i] * dzda[n][j]
          else
            proj[n][i][j] =     - dzda[n][i] * dzda[n][j]
          end
        end
      end
    end
    return proj
  end
end
