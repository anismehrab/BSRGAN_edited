import onnx
import os
import onnxruntime
import numpy as np
from onnx_tf.backend import prepare
import torch
from models.network_rrdbnet import RRDBNet as net
import tensorflow as tf
import time
from torch.utils.mobile_optimizer import optimize_for_mobile


















torch_model_scripted_vulkan_path = os.path.join('model_zoo', 'BSRGAN_scripted_vulkan.pt')
torch_model_scripted_path = os.path.join('model_zoo', 'BSRGAN_scripted.pt')
torch_model_path = os.path.join('model_zoo', 'BSRGAN.pth') 
torch_model_quant_path = os.path.join('model_zoo', 'BSRGAN_quant.pt') 
torch_model_quant_scripted_path = os.path.join('model_zoo', 'BSRGAN_quant_scripted.pt') 
torch_model_quant_scripted_vulakn_path = os.path.join('model_zoo', 'BSRGAN_quant_scripted_vulkan.pt') 
onnx_path = os.path.join('model_zoo', 'BSRGAN_ONNX.onnx')        
onnx_quant_path = os.path.join('model_zoo', 'BSRGAN_ONNX_scripted.onnx')        
tf_rep_path = os.path.join('model_zoo', 'BSRGAN_tf_Rep')         
tf_lite_path = os.path.join('model_zoo', 'BSRGAN_tf_Lite')         
device = torch.device('cpu')#('cuda' if torch.cuda.is_available() else 'cpu')








#LOAD TORCH MODEL
torch_model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
torch_model.load_state_dict(torch.load(torch_model_path), strict=True)
torch_model.eval()
torch_model = torch_model.to(device)
torch.cuda.empty_cache()


#Post Training Static Quantization
# backend = "qnnpack"
# torch_model.qconfig = torch.quantization.get_default_qconfig(backend)
# torch.backends.quantized.engine = backend
# model_static_quantized = torch.quantization.prepare(torch_model, inplace=False)
# model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
#torch.save(model_static_quantized.state_dict(), torch_model_quant_path)

#TORCHSCRIPT
scripted_model = torch.jit.script(torch_model)
scripted_model_optimized = optimize_for_mobile(scripted_model,backend='cpu')
scripted_model_optimized._save_for_lite_interpreter(torch_model_scripted_path)


#TORCH TO ONNX

# def exportToOnnx(model,input,output,path):
#     print("export the torch model to "+path)
#     # Export the model
#     torch.onnx.export(model,               # model being run
#                         input,                         # model input (or a tuple for multiple inputs)
#                         path,   # where to save the model (can be a file or file-like object)
#                         example_outputs=output,
#                         export_params=True,        # store the trained parameter weights inside the model file
#                         opset_version=12,          # the ONNX version to export the model to
#                         do_constant_folding=True,  # whether to execute constant folding for optimization
#                         input_names = ['input'],
#                         output_names = ['output'],
#                         dynamic_axes={
#                                     'input' : {2 : 'inputc_h', 3: 'inputc_w'},
#                                     'output' : {2 : 'output_h', 3: 'output_w'},
#                                     }
#                                     );
#     print("finished exporting to onnx")

# print("mode loaded to {}".format(device))
# img_L = torch.from_numpy(np.random.randn(1, 3, 50, 50).astype(np.float32))
# img_E = torch_model(img_L.to(device))
# exportToOnnx(model = torch_model,input = img_L,output = img_E,path = onnx_path)





# # Load the ONNX model
# model_onnx = onnx.load(onnx_path)

# #Check that the IR is well formed
# onnx.checker.check_model(model_onnx)

# #Print a Human readable representation of the graph
# print(onnx.helper.printable_graph(model_onnx.graph))

# ort_session = onnxruntime.InferenceSession(onnx_path)

# outputs = ort_session.run(
#     None,
#     {'input': np.random.randn(1, 3, 80, 80).astype(np.float32)}
# )
# print(np.shape(outputs))



# #ONNX to TF
# tf_rep = prepare(model_onnx)    
# tf_rep.export_graph(tf_rep_path)



# #TF Model Inference
# model_tf = tf.saved_model.load(tf_rep_path)
# model_tf.trainable = False

# # input_tensor = tf.random.uniform([1, 3, 40, 40])
# # out = model_tf(**{'input': input_tensor})
# # print(out["output"].shape)




# #TF to TFLite
# # Convert the model
# converter = tf.lite.TFLiteConverter.from_saved_model(tf_rep_path)
# tflite_model = converter.convert()

# # Save the model
# with open(tf_lite_path+".tflite", 'wb') as f:
#     f.write(tflite_model)


# #optimization
# converter = tf.lite.TFLiteConverter.from_saved_model(tf_rep_path)
# #Convert using dynamic range quantization
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# #Convert using integer-only quantization
# #Ensure that if any ops can't be quantized, the converter throws an error
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# # Set the input and output tensors to uint8 (APIs added in r2.3)
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# tflite_quant_model = converter.convert()
# # Save the quantizated model
# with open(tf_lite_path+"_quant.tflite", 'wb') as f:
#     f.write(tflite_quant_model)

#TFLite Model Inference
# Load the TFLite model and allocate tensors
# interpreter = tf.lite.Interpreter(model_path=tf_lite_path)

# input_details = interpreter.get_input_details()

# #resize inputs to match model inputs
# interpreter.resize_tensor_input(
#     input_details[0]['index'], (1, 3, 150, 150))
# interpreter.allocate_tensors()
# # Get input and output tensors
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# input_shape = input_details[0]['shape']
# print(input_shape)
# # Test the model on random input data

# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

# interpreter.allocate_tensors()
# interpreter.set_tensor(input_details[0]['index'], input_data)

# interpreter.invoke()

# # get_tensor() returns a copy of the tensor data
# # use tensor() in order to get a pointer to the tensor
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(np.shape(output_data))






