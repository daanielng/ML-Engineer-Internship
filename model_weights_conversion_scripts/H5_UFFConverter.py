#!/usr/bin/env python
"""
python keras_to_tensorflow.py 
    --input_model="path/to/keras/model.h5" 
    --output_model="path/to/save/model.pb"
"""

import tensorflow as tf
from tensorflow.compat.v1 import graph_util
#from tensorflow.python.keras import backend as K
from tensorflow.keras import backend as K
import argparse
from pathlib import Path
import pprint
import sys
import uff
import sys

#necessary !!!


def config_generation(model , input_layer_name , output_node_name,model_path):
    height = str(int(model.input.shape[1]))
    width = str(int(model.input.shape[2]))
    channel = str(int(model.input.shape[3]))
    input_layer_name = input_layer_name
    output_layer_name = output_node_name
    
    model_name = model_path[0:-2]+"uff"
    lable_name = model_path[0:-2]+"txt"
    precision = "fp32"
    type_channel="RGB"
    batch="4"
    norm="1"
    bias ="0"
    config_param = {"model":model_name,
                    "label":lable_name,
                    "precision":precision,
                    "input":input_layer_name,
                    "output":output_layer_name,
                    "channel":channel,
                    "width":width,
                    "height":height,
                    "type":type_channel,
                    "batch":batch,
                    "norm":norm,
                    "bias":bias,
                    "yoloLabel":"person"
                    }
    
    
    file = open(model_path[0:-2]+"config","w")
    count = 0
    for key in config_param:
        print(key+"="+config_param[key])
        file.write(key+"="+config_param[key])
        if(count !=(len(config_param)-1)):
            file.write("\n")
        count+=1
    file.close()




def convertpbtouff(filename,output_node):
    print('Converting...')
   
    output_filename = filename[:filename.rfind('.')]  + '.uff'
  
    trt_graph = uff.from_tensorflow_frozen_model(filename, output_nodes=[output_node])
    print('Done')
    print('Writing to disk...')
    with open(output_filename, 'wb') as f:
        f.write(trt_graph)
    print('Done')





tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
req_grp = parser.add_argument_group('required')
req_grp.add_argument('--input_model',   default=False, type=str, help='Path to the input model.')
req_grp.add_argument('--output_model',  default=False, type=str, help='Path where the converted model will.')

args = parser.parse_args()
pprint.pprint(vars(args))

# input model path
model_path = args.input_model

# If output_model path is relative and in cwd, make it absolute from root
output_model = args.output_model
out_path = output_model
if str(Path(output_model).parent) == '.':
    output_model = str((Path.cwd() / output_model))

output_fld = Path(output_model).parent
output_model_name = Path(output_model).name

K.set_learning_phase(0)

restored_model = tf.keras.models.load_model(model_path)



print(restored_model.outputs)
print(restored_model.inputs)

output_layer_name = [node.name for node in restored_model.outputs]
output_layer_name = output_layer_name[0][0:-2]
inputLayerName = [node.name for node in restored_model.inputs]
inputLayerName =  inputLayerName[0][0:-2]

print(output_layer_name)
print(inputLayerName)

restored_model.summary()

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

frozen_graph = freeze_session(tf.compat.v1.keras.backend.get_session(),
                              output_names=[out.op.name for out in restored_model.outputs],
                              clear_devices=True)

tf.io.write_graph(frozen_graph, str(output_fld), output_model_name, as_text=False)
print("save pb successfully! ")
convertpbtouff(output_model_name,output_layer_name)
config_generation(restored_model,inputLayerName,output_layer_name,out_path)