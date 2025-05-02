#include "trt_infer.hpp"
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>

TRTInfer::TRTInfer(const std::string& engine_path)
{
   
    nvinfer1::IRuntime* runtime =
        nvinfer1::createInferRuntime(nvinfer1::ILogger::Severity::kWARNING);
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file) throw std::runtime_error("Engine file não encontrado");

    engine_file.seekg(0, std::ios::end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(), engine_size);

    engine = runtime->deserializeCudaEngine(engine_data.data(), engine_size);
    context = engine->createExecutionContext();

    inputIndex = engine->getBindingIndex("input");
    outputIndex = engine->getBindingIndex("output");

    auto input_dims = engine->getBindingDimensions(inputIndex);
    batchSize = input_dims.d[0];
    inputH = input_dims.d[2];
    inputW = input_dims.d[3];


    size_t input_size = 1 * 3 * inputH * inputW * sizeof(float);
    size_t output_size = 1 * 1 * inputH * inputW * sizeof(float);
    cudaMalloc(&buffers[inputIndex], input_size);
    cudaMalloc(&buffers[outputIndex], output_size);

    cudaStreamCreate(&stream);

    std::cout << "[INFO] Engine carregada com sucesso.\n";
}

TRTInfer::~TRTInfer()
{
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    context->destroy();
    engine->destroy();
}

void TRTInfer::preprocess(const cv::Mat& img, float* gpu_input)
{
    cv::Mat resized, rgb;
    cv::resize(img, resized, cv::Size(inputW, inputH));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    float host_input[3 * inputH * inputW];
    int idx = 0;
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < inputH; ++y)
            for (int x = 0; x < inputW; ++x)
                host_input[idx++] = rgb.at<cv::Vec3b>(y, x)[c] / 255.0f;

    cudaMemcpyAsync(buffers[inputIndex], host_input, sizeof(host_input),
                    cudaMemcpyHostToDevice, stream);
}

cv::Mat TRTInfer::postprocess(float* gpu_output)
{
    float host_output[inputH * inputW];
    cudaMemcpyAsync(host_output, buffers[outputIndex], sizeof(host_output),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cv::Mat mask(inputH, inputW, CV_8UC1);
    for (int y = 0; y < inputH; ++y)
        for (int x = 0; x < inputW; ++x)
            mask.at<uchar>(y, x) = host_output[y * inputW + x] > 0.5 ? 255 : 0;

    return mask;
}

cv::Mat TRTInfer::infer(const cv::Mat& input)
{
    preprocess(input, (float*)buffers[inputIndex]);
    context->enqueueV2(buffers, stream, nullptr);
    return postprocess((float*)buffers[outputIndex]);
}
