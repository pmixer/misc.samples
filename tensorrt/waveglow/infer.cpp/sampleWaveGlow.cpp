// on loading the engine file: https://devtalk.nvidia.com/default/topic/1030042/loading-of-the-tensorrt-engine-in-c-api/?offset=20

#include "logger.h"
#include "common.h"
#include "buffers.h"
#include "NvInfer.h"
#include "NvUtils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

// using namespace nvinfer1;

class SampleWaveGlow
{
public:
    SampleWaveGlow(char *engine_fname)
    {
        // load the plan file and create engine
        std::stringstream modelBuff;
        modelBuff.seekg(0, modelBuff.beg);
        std::ifstream cache(engine_fname);
        modelBuff << cache.rdbuf();
        cache.close();

        this->runtime = nvinfer1::createInferRuntime(gLogger);
        
        modelBuff.seekg(0, std::ios::end);
        const int modelSize = modelBuff.tellg();
        modelBuff.seekg(0, std::ios::beg);
        void* modelMem = malloc(modelSize);
        modelBuff.read((char*)modelMem, modelSize);
        
        this->engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelMem, modelSize, NULL), samplesCommon::InferDeleter());
        
        free(modelMem);

        this->context = this->engine->createExecutionContext();

        int batchSize = 1;
        this->buffers = new samplesCommon::BufferManager(this->engine, batchSize);

        string in_node_name = "data", out_node_name = "(Unnamed Layer* 1355) [Shuffle]_output";
        this->hostin = (float *)this->buffers->getHostBuffer(in_node_name);
        this->hostout = (float *)this->buffers->getHostBuffer(out_node_name);
    }

    ~SampleWaveGlow()
    {
        this->context->destroy();
        this->runtime->destroy();
        // this->engine->destroy(); // destroy the shared_ptr?
    }

    void infer()
    {
        cudaStream_t stream; cudaStreamCreate(&stream);
        auto t_start = std::chrono::high_resolution_clock::now();
        this->buffers->copyInputToDeviceAsync(stream); int bz = 1; // batch size
        context->enqueue(bz, this->buffers->getDeviceBindings().data(), stream, nullptr);
        this->buffers->copyOutputToHostAsync(stream);

        cudaStreamSynchronize(stream);
        auto t_end = std::chrono::high_resolution_clock::now();

        cudaStreamDestroy(stream); auto time_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();
        std::cout << "Inference(H->D->H) took " << time_in_ms << " milliseconds. ";
        int buff_size_fp32 = this->buffers->size("(Unnamed Layer* 1355) [Shuffle]_output") / 4;
        std::cout << "RTF = " << time_in_ms * 22050. / (buff_size_fp32 * 1000) << std::endl;
    }


    float * hostin = nullptr, * hostout = nullptr;
    samplesCommon::BufferManager *buffers = nullptr;

private:
    nvinfer1::IRuntime *runtime = nullptr;
    shared_ptr<nvinfer1::ICudaEngine> engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;

};

int main(int argc, char** argv)
{
    assert(argc >= 2); char *engine_fname = argv[1]; 
    SampleWaveGlow vocoder(engine_fname);

    // feedin input
    ifstream cond_in("cond.in"); vector<float> nums; float tmp;
    while (cond_in >> tmp) {nums.push_back(tmp);}
    int cond_size = nums.size(); 
    int buff_size_fp32 = vocoder.buffers->size("data") / 4;
    int expected_len = buff_size_fp32 / 80, cond_len = cond_size / 80;
    int min_len = std::min(expected_len, cond_len);

    for (int i = 0; i < 80; i++) {
        for (int j = 0; j < expected_len; j++)
        {
            if (j < min_len)
                vocoder.hostin[i * expected_len + j] = nums[i * cond_len + j];
            else
                vocoder.hostin[i * expected_len + j] = 0;
        }
    }

    // inference
    vocoder.infer();

    // dump output
    ofstream audio_out("audio.out");
    buff_size_fp32 = vocoder.buffers->size("(Unnamed Layer* 1355) [Shuffle]_output") / 4;
    for (int i = 0; i < buff_size_fp32; i++) {
        audio_out << vocoder.hostout[i] << std::endl;
    }
    return 0;
}
