import copy

from src.gnn.gnn import GNN
from src.parser.parser import Parser
from src.parser.onnxtonn import ONNX
from src.utilities.log import Log
from src.utilities.spec import Spec
from src.utilities.vnnlib import VNNLib

##################################
##### Initiate the log file ######
##################################
f = open("log.txt", "w")
f.close()
#####################################
##### Reading network and spec ######
#####################################
# Network file
filePath: str = \
    "/Users/ratanlal/Desktop/repositories/github/VERINN/resources/acasxuonnx/ACASXU_run2a_1_1_batch_2000.onnx"
# Property file
specPath: str = (
        "/Users/ratanlal/Desktop/repositories/github/VERINN/resources/acasxuvnnlib/prop_" +
        str(1) + ".vnnlib")
# Parse data from the network file
objParser: Parser = ONNX(filePath)
# create an instance of GNN
objGNN: GNN = objParser.getNetwork()
# Display network structure
Log.message("Original Network Structure\n")
objGNN.display()

# Parse specification from the property file
i, o, d = VNNLib.get_num_inputs_outputs(filePath)
ioSpec = VNNLib.read_vnnlib_simple(specPath, i, o)
objInputSet = Spec.getInput(ioSpec)
Log.message(objInputSet.display())
outputConstr = Spec.getOutput(ioSpec)
tempInputSet = copy.deepcopy(objInputSet)