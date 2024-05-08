##
#  Path Optimization with String Method
##

require_relative 'constants'
require_relative 'molecule'

##
#  QM Engine Class
##

class QMEngine
  attr_accessor :engine,
                :reptag,
                :filename,
                :scratch,
                :node,
                :nodelog,
                :ncpus
  def initialize 
    @reptag  = "___"
    @filename = {
      input:  "node.inp",
      output: "node.out",
      punch:  "node.punch"
    }
    @scratch = "/scratch"
    @node = []
    @nodelog = "node.log"
    @hostname = []
    @ncpus = 1
  end
  def make_inp(filename, path)
    template = File.read(filename)
    node.each_with_index do |node, n|
      filename = "#{node}/#{@filename[:input]}"
      File.open(filename, "w") do |fw|
        fw.puts template.gsub(/#{@reptag}/, path.mol[n].print(datatype: :geom, unitconv: true, shownatoms: false))
      end
    end
  end
end
