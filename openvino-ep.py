import onnxruntime
import os
import os.path
import sys
import numpy
import time
import argparse

parser = argparse.ArgumentParser(description='Using OpenVINO Execution Provider for ONNXRuntime')
parser.add_argument('--device', default='cpu', help="Device to perform inference on 'cpu (MLAS)' or on devices supported by OpenVINO-EP [CPU_FP32, GPU_FP32, GPU_FP16, MYRIAD_FP16, VAD-M_FP16].")
parser.add_argument('--iters', help = "Number of iterations")
args = parser.parse_args()

onnxruntime.set_default_logger_severity(0) #Prints additional logger prints for easy debugging

sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 1
model_path="bvlc_alexnet/model.onnx"
sess = onnxruntime.InferenceSession(model_path, sess_options)
print("\n")
print("Printing session providers: ")
print("\n")
print(sess.get_providers())
print("\n")



device = args.device

if(args.device == 'cpu'):
    print("Device type selected is 'cpu' which is the default CPU Execution Provider (MLAS)")
else:
    # Set OpenVINO as the Execution provider to infer this model
    sess.set_providers(['OpenVINOExecutionProvider'], [{'device_type' : device}])
    print("Device type selected is: {} using the OpenVINO Execution Provider".format(device))
    '''
    other 'device_type' options are: (Any hardware target can be assigned if you have the access to it)
    'CPU_FP32', 'GPU_FP32', 'GPU_FP16', 'MYRIAD_FP16', 'VAD-M_FP16'
    '''



print("Inputs info: ")
for elem in sess.get_inputs():
    print(elem)
print("\n")
    
print("Outputs info: ")
for elem in sess.get_outputs():
    print(elem)
print("\n")

input_shape = sess.get_inputs()[0].shape
print("Model input shape: ", input_shape)

x = numpy.random.random(input_shape).astype(numpy.float32)
iters = int(args.iters)
ticks = time.time()
for i in range(iters): # 1 for running just 1 iteration
    res = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: x})
ticks = time.time() - ticks
print("Inference time per frame: ", ticks/iters)
out = numpy.array(res)

#print("output: ", out)
print("Output probabilities:", out.shape)
