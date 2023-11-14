import onnx
from onnx import helper
import sys, getopt
import base64

class Params:
    def __init__(self):
        self.class_ids = []
        self.class_nodeids = []
        self.class_treeids = []
        self.class_weights = []
        self.classlabels = []
        self.nodes_falsenodeids = []
        self.nodes_featureids = []
        self.nodes_hitrates = []
        self.nodes_missing_value_tracks_true = []
        self.nodes_modes = []
        self.nodes_nodeids = []
        self.nodes_treeids = []
        self.nodes_truenodeids = []
        self.nodes_values = []
        self.base_values =  []
        self.post_transform =  "SOFTMAX"

    def DataStore(self, name, data):
        if name == "class_ids":
            self.class_ids = data
        if name == "class_nodeids":
            self.class_nodeids = data
        if name == "class_treeids":
            self.class_treeids = data
        if name == "class_weights":
            self.class_weights = data
        if name == "classlabels_int64s":
            self.classlabels = data
        if name == "nodes_falsenodeids":
            self.nodes_falsenodeids = data
        if name == "nodes_featureids":
            self.nodes_featureids = data
        if name == "nodes_hitrates":
            self.base_values = data
        if name == "nodes_missing_value_tracks_true":
            self.nodes_missing_value_tracks_true = data
        if name == "nodes_modes":
            temp = []
            for ele in data:
                s = str(ele)
                temp.append(s[2:-1])
            self.nodes_modes = temp
        if name == "nodes_nodeids":
            self.nodes_nodeids = data
        if name == "nodes_treeids":
            self.nodes_treeids = data
        if name == "nodes_truenodeids":
            self.nodes_truenodeids = data
        if name == "nodes_values":
            self.nodes_values = data
        if name == "base_values":
            self.base_values = data
        if name == "post_transform":
            self.post_transform = data[2:-1].decode('utf-8').replace("'", "\"")

    def Output(self):
        return f"""params = {"{"}
    "class_ids": {self.class_ids},
    "class_nodeids": {self.class_nodeids},
    "class_treeids": {self.class_treeids},
    "class_weights": {self.class_weights},
    "classlabels": {self.classlabels},
    "nodes_falsenodeids": {self.nodes_falsenodeids},
    "nodes_featureids": {self.nodes_featureids},
    "nodes_hitrates": {self.nodes_hitrates},
    "nodes_missing_value_tracks_true": {self.nodes_missing_value_tracks_true},
    "nodes_modes": {self.nodes_modes},
    "nodes_nodeids": {self.nodes_nodeids},
    "nodes_treeids": {self.nodes_treeids},
    "nodes_truenodeids": {self.nodes_truenodeids},
    "nodes_values": {self.nodes_values},
    "base_values": {self.base_values},
    "post_transform": "SOFTMAX",
{"}"}"""

def match_type(index):
    types =     ["UNDEFINED",
                "FLOAT",
                "INT",
                "STRING",
                "TENSOR",
                "GRAPH",
                "FLOATS",
                "INTS",
                "STRINGS",
                "TENSORS",
                "GRAPHS",
                "SPARSE_TENSOR",
                "SPARSE_TENSORS",
                "TYPE_PROTO",
                "TYPE_PROTOS"]
    if index > len(types):
        return "ErrorIndex"
    else:
        return types[index]


# Load onnx model
def loadOnnxModel(path):
    model = onnx.load(path)
    return model


# Get a list of input and output names for nodes and nodes
def getNodeAndIOname(nodename,model):
    for i in range(len(model.graph.node)):
        if model.graph.node[i].name == nodename:
            Node = model.graph.node[i]
            input_name = model.graph.node[i].input
            output_name = model.graph.node[i].output
    return Node,input_name,output_name

def getAttributeList(attr, attr_type):
    if attr_type == "floats":
        return attr.floats
    if attr_type == "string":
        return attr.s
    if attr_type == "ints":
        return attr.ints
    if attr_type == "strings":
        return attr.strings

DataSet = ["Acute_Inflammations", "Breast_Cancer", "Chronic_Kidney_Disease", "Heart_Disease", "Heart_Failure_Clinical_Records", "Lymphography", "Parkinsons"]
for data in DataSet:
    model = loadOnnxModel(f'{data}/{data}.onnx')
    Node,input_name,output_name = getNodeAndIOname("TreeEnsembleClassifier", model)
    params = Params()
    for attr in Node.attribute:
        attr_type = str.lower(match_type(attr.type))
        attributeList = getAttributeList(attr, attr_type)
        params.DataStore(attr.name, attributeList)

    with open(f"{data}/params.txt", 'w') as file:
        file.write(params.Output())
