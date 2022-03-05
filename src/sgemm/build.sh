export CUDA_VISIBLE_DEVICES="0"
nvcc -arch sm_60 sgemm.cu -o sgemm --ptxas-options=-v 
