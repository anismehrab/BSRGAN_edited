# import onnx
# import onnxruntime
# from onnx_tf.backend import prepare
# from onnxruntime.quantization import quantize
# from onnxruntime.quantization.quant_utils import QuantizationMode

# import tensorflow as tf

import os
import numpy as np
import torch
from models.network_rrdbnet_scripted import RRDBNet as net
import time
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.backends._nnapi import prepare as nnapi_prepare
import torch.quantization.quantize_fx as quantize_fx
import copy

from utils import utils_image as util
import cv2















torch_model_scripted_vulkan_path = os.path.join('model_zoo', 'BSRGAN_scripted_vulkan.pt')
torch_model_scripted_path = os.path.join('model_zoo', 'BSRGAN_scripted.pt')
torch_model_path = os.path.join('model_zoo', 'BSRGAN.pth') 
torch_model_quant_path = os.path.join('model_zoo', 'BSRGAN_quant.pt') 
torch_model_quant_scripted_path = os.path.join('model_zoo', 'BSRGAN_quant_scripted.pt') 
torch_model_scripted_nnapi_path = os.path.join('model_zoo', 'BSRGAN_scripted_nnapi.pt') 
torch_model_quant_scripted_vulakn_path = os.path.join('model_zoo', 'BSRGAN_quant_scripted_vulkan.pt') 
onnx_path = os.path.join('model_zoo', 'BSRGAN_ONNX.onnx')        
onnx_quant_path = os.path.join('model_zoo', 'BSRGAN_ONNX_quant.onnx')        
onnx_quant_static_path = os.path.join('model_zoo', 'BSRGAN_ONNX_quant_static.onnx')        
tf_rep_path = os.path.join('model_zoo', 'BSRGAN_tf_Rep')         
tf_lite_path = os.path.join('model_zoo', 'BSRGAN_tf_Lite')         
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')








#LOAD TORCH MODEL
torch_model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
torch_model.load_state_dict(torch.load(torch_model_path))
torch_model.eval()
#torch_model = torch_model.to(device)

#Post Training Static Quantization
# modules_to_fuse = [['upconv1', 'lrelu'],
#                    ['upconv2', 'lrelu'],
#                     ['HRconv', 'lrelu'],
#                  ]
# model_f32_fused = torch.quantization.fuse_modules(torch_model, modules_to_fuse, inplace=True)
torch_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
model_fp32_prepared =torch.quantization.prepare(torch_model)

model_fp32_prepared.to(device)
L_path = os.path.join("testsets", 'RealSRSet')

with torch.no_grad():
    for img in util.get_image_paths(L_path):
        torch.cuda.empty_cache()

        img_L = util.imread_uint(img, n_channels=3)#return numpy array RGB H*W*C
        if(np.shape(img_L)[0] < 512 and np.shape(img_L)[1] < 512):
            img_L = util.uint2tensor4(img_L)#return pytorch tensor with 1*C*H*W
            
            img_L = img_L.to(device)
            model_fp32_prepared(img_L)
            
            
        
model_fp32_prepared.eval()    
model_fp32_prepared.cpu()        
model_int8_quantized =torch.quantization.convert(model_fp32_prepared)
input_tensor = torch.from_numpy(np.random.randn(1, 3, 50, 50).astype(np.float32))
print(model_int8_quantized(input_tensor))
torch.save(model_int8_quantized.state_dict(), torch_model_quant_path)












#FX Graph Mode Quantization
# torch_model.to(device)
# model_to_quantize = copy.deepcopy(torch_model)
# qconfig_dict = {"": torch.quantization.get_default_qconfig('qnnpack')}
# model_to_quantize.eval()
# # prepare
# model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_dict)
# # calibrate (not shown)
# model_prepared.to(device)
# L_path = os.path.join("testsets", 'RealSRSet')

# with torch.no_grad():
#     for img in util.get_image_paths(L_path):
#         torch.cuda.empty_cache()

#         img_L = util.imread_uint(img, n_channels=3)#return numpy array RGB H*W*C
#         if(np.shape(img_L)[0] < 512 and np.shape(img_L)[1] < 512):
#             img_L = util.uint2tensor4(img_L)#return pytorch tensor with 1*C*H*W
            
#             img_L = img_L.to(device)
#             model_prepared(img_L)
#             torch.cuda.empty_cache()
# # quantize
# model_quantized = quantize_fx.convert_fx(model_prepared)
# input_tensor = torch.from_numpy(np.random.randn(1, 3, 50, 50).astype(np.float32))
# print(model_quantized(input_tensor))
# torch.save(model_quantized.state_dict(), torch_model_quant_path)
# quantized_MODEL = torch.quantization.quantize_dynamic(
#     torch_model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
# )

# #dynamic/weight_only Quantization
# model_to_quantize = copy.deepcopy(torch_model)
# model_to_quantize.eval()
# qconfig_dict = {"": torch.quantization.default_dynamic_qconfig}
# # prepare
# model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_dict)
# # no calibration needed when we only have dynamici/weight_only quantization
# # quantize
# model_quantized = quantize_fx.convert_fx(model_prepared)


#TORCHSCRIPT
scripted_model = torch.jit.script(model_int8_quantized)
scripted_model_optimized = optimize_for_mobile(scripted_model,backend="cpu")
scripted_model_optimized._save_for_lite_interpreter(torch_model_quant_scripted_path)
scripted_model_optimized = optimize_for_mobile(scripted_model,backend="Vulkan")
scripted_model_optimized._save_for_lite_interpreter(torch_model_quant_scripted_vulakn_path)

# #to NNAPI 
# scripted_model = torch.jit.script(model_int8_quantized)
# input_tensor = torch.from_numpy(np.random.randn(1, 3, 50, 50).astype(np.float32))
# input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
# input_tensor.nnapi_nhwc = True
# nnapi_model = nnapi_prepare.convert_model_to_nnapi(scripted_model,input_tensor)
# nnapi_model._save_for_lite_interpreter(torch_model_scripted_nnapi_path)
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
# img_L = torch.from_numpy(np.random.randn(1, 3, 50, 50).astype(np.uint8))
# img_E = torch.from_numpy(np.random.randn(1, 3, 200, 200).astype(np.uint8))
# exportToOnnx(model = model_quantized,input = img_L,output = img_E,path = onnx_path)





# # Load the ONNX model
# model_onnx = onnx.load(onnx_path)

# # #Check that the IR is well formed
# # onnx.checker.check_model(model_onnx)
# # #Print a Human readable representation of the graph
# # print(onnx.helper.printable_graph(model_onnx.graph))
# quantized_model = quantize(model_onnx,
#                             quantization_mode=QuantizationMode.IntegerOps,
#                             symmetric_weight=True,
#                             force_fusions=True)

# onnx.save(quantized_model, onnx_quant_path)


# ort_session = onnxruntime.InferenceSession(onnx_quant_path)

# outputs = ort_session.run(
#     None,
#     {'input': np.random.randn(1, 3, 80, 80).astype(np.float32)}
# )
# print(np.shape(outputs))



#ONNX to TF
# model_onnx = onnx.load(onnx_quant_path)
# tf_rep = prepare(model_onnx)    
# tf_rep.export_graph(tf_rep_path)



# #TF Model Inference
# model_tf = tf.saved_model.load(tf_rep_path)
# model_tf.trainable = False

# input_tensor = tf.random.uniform([1, 3, 40, 40])
# out = model_tf(**{'input': input_tensor})
# print(out["output"].shape)




#TF to TFLite
# Convert the model
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
# # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# # # Set the input and output tensors to uint8 (APIs added in r2.3)
# # converter.inference_input_type = tf.uint8
# # converter.inference_output_type = tf.uint8
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






