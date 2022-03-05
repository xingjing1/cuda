export CUDA_VISIBLE_DEVICES="1"
nvcc -arch sm_60 batch.cu -o batch --ptxas-options=-v 
