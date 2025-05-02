#ifndef TRT_INFER_HPP
#define TRT_INFER_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

class TRTInfer {
public:
    TRTInfer(const std::string& engine_path);
    ~TRTInfer();

 
    cv::Mat infer(const cv::Mat& input);

private:
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    void* buffers[2];
    int inputIndex, outputIndex;

    cudaStream_t stream;

    int batchSize, inputH, inputW;

    void preprocess(const cv::Mat& img, float* gpu_input);
    cv::Mat postprocess(float* gpu_output);
};

#endif
