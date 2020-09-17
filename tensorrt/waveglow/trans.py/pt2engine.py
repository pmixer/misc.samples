# basics
import sys, os, json, numpy as np

# TRT-py rely on
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# pytorch implementation of WG
import torch, glow

# to infer, write audio and record speed
from scipy.io.wavfile import write
from time import time
import trt_common

# if INT8 ON
import calib

N, C, H, W = -1, -1, -1, -1

class WaveGlow:
    def __init__(self, model_fname='waveglow_256channels.pt'):

        # load hyper parameters
        with open('hparams.json') as f:
            self.hps = json.load(f)
        global N, C, H, W
        N, C, H, W = [self.hps['input'][k] for k in 'NCHW']

        # load PyTorch model weights
        self.model = torch.load(model_fname, map_location='cpu')['model']
        self.model = self.model.remove_weightnorm(self.model)
        self.weights_dict = self.model.state_dict()
        
        # just add all env info to generated engine name
        s = model_fname.split('.')[0]
        s += '.gpu.' + pycuda.driver.Device(0).name().replace(" ", '-')
        s += ".trt." +  trt.__version__ + '.workspace.'
        s += str(self.hps['tensorrt']['max_workspace_size']) + "GiB"
        cv = str(pycuda.driver.get_version())[1:-1]
        cv = cv.replace(" ", '').replace(",",".")
        dv = str(pycuda.driver.get_driver_version())
        self.fname_header = s + '.cuda.' + cv + '.driver.' + dv

        precision = '.fp32'
        if self.hps['tensorrt']['FP16_ON']: precision = ".fp16"
        if self.hps['tensorrt']['INT8_ON']: precision = ".int8"
        self.fname_header += precision

        self.fname_header += '.N.C.H.W.' + '.'.join([str(self.hps['input'][k]) for k in 'NCHW'])

        # set TrT builder and get network handle
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        self.builder = trt.Builder(self.logger)
        self.builder.debug_sync = False
        self.builder.fp16_mode = self.hps['tensorrt']['FP16_ON']
        if self.hps['tensorrt']['INT8_ON']:
            self.int8_mode = True
            cn = self.fname_header + '.cache'
            worker = calib.WaveGlowEntropyCalibrator('pts/', cn, C, W)
            self.builder.int8_calibrator = worker
        self.builder.max_batch_size = self.hps['input']['N']
        GiB = self.hps['tensorrt']['max_workspace_size'] << 30
        self.builder.max_workspace_size = GiB 
        
        bitmask = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.nw = self.builder.create_network(bitmask) # handle to add layer
        
        # to enable dynamic shape input feature in EXPLICIT_BATCH mode
        profile = self.builder.create_optimization_profile() # min, opt, max
        profile.set_shape("spect", (N,C,1,80), (N,C,1,800), (N,C,1,4800))
        config = self.builder.create_builder_config()
        config.add_optimization_profile(profile)

        # intermidiate parameters for building the compute graph
        self.dT = self.hps['input']['W'] * 256 // self.hps['model']['n_group']

        # increase channels num at flow 8, 4
        self.nrc = 4 # n_remaining_channels
        self.nes = self.hps['model']['n_early_size']


    def destory(self):
        '''
        destroy self.builder etc. to free resources
        '''
        pass # TODO, make everything clean when exit


    def build_engine(self):
        '''
        generate and save TrT engine after graph built
        '''
        engine = self.builder.build_cuda_engine(self.nw)
        engine_fname = self.fname_header + '.engine'
        engine_path = os.path.join('engines', engine_fname)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
            print("TRT Engine saved to: " + engine_path)


    def infer(self, engine, mel='pts/LJ001-0051.wav.pt', use_trt=True):
        '''
        inference in tensorrt/pytorch, take mel to generate audio
        '''
        MAX_WAV_VALUE = 32768.0
        sampling_rate = self.hps['model']['sampling_rate']
        if use_trt:
            iter_num = 10 # number of iterations after the warmup round
            mel = torch.load(mel).numpy()
            with open(engine, 'rb') as f, trt.Runtime(self.logger) as rt:
                engine = rt.deserialize_cuda_engine(f.read())
            i, o, b, s = trt_common.allocate_buffers(engine)
            ctx = engine.create_execution_context()
            ctx.set_binding_shape(0, (1, 80, 1, 800))
            ctx.active_optimization_profile = 0
            bsz = engine.max_batch_size
            audio_len = len(i[0].host) // bsz
            mel_buff = np.zeros((80, audio_len // 80))
            idx = min(audio_len // 80, mel.shape[1])
            mel_buff[:, :idx] = mel[:, :idx]
            mel_in = mel_buff.flatten()
            mel_in = np.tile(mel_in, bsz)
            if len(i) > 0: np.copyto(i[0].host, mel_in)

            for iter_idx in range(iter_num + 1):
                res_list, _ = trt_common.do_inference(ctx, b, i, o, s, bsz)
                s.synchronize()
                if iter_idx == 0: start = time()

            end = time()
            out = res_list[-1]
            seconds = (len(out)/sampling_rate)
            rtf = (end - start) / iter_num / seconds
            print('Gen {} seconds audio, RTF={}'.format(seconds, rtf))
            audio = out * MAX_WAV_VALUE
            audio = audio.astype('int16')
            write('trt.wav', sampling_rate, audio)
        else:
            mel = torch.autograd.Variable(torch.load(mel)).unsqueeze(0)
            with torch.no_grad():
                out = self.model.forward(mel)
                audio = out * MAX_WAV_VALUE
            audio = audio.squeeze().numpy().astype('int16')
            write('pytorch.wav', sampling_rate, audio) # scipy func


    def get_t(self, tensor_name, reverse=False):
        '''
        get weight/bias tensor by its key in weights_dict
        '''
        tensor = self.weights_dict[tensor_name]
        if reverse:
          tensor = tensor.squeeze(-1).inverse()
        return trt.Weights(tensor.numpy().flatten())


    def build_graph(self):
        '''
        Rebuild https://github.com/NVIDIA/waveglow in TrT.
        '''
        global N, C, H, W
        spect = self.nw.add_input('spect', trt.float32, (N,C,H,W))

        upsw, upsb = self.get_t('upsample.weight'), self.get_t('upsample.bias')
        L = self.nw.add_deconvolution(spect, C, (1, 1024), upsw, upsb) # TODO
        L.name, L.stride = "up_tconv", (1, 256) # smarter way for layer names?

        L = self.nw.add_slice(L.get_output(0),(0,0,0,0),(N,C,1,W*256),(1,1,1,1))
        L.name = "up_slice"

        L = self.nw.add_shuffle(L.get_output(0)); L.name = "up_shuffle_1"
        ng = self.hps['model']['n_group']
        L.reshape_dims, L.second_transpose = (N,C,self.dT,ng), (0,2,1,3)

        c = self.nw.add_shuffle(L.get_output(0)); c.name = "up_shuffle_2"
        dC = self.hps['model']['n_group'] * self.hps['input']['C']
        c.reshape_dims, c.second_transpose = (N,self.dT,1,dC), (0,3,2,1)

        arand = np.random.normal(0., 1., (N,self.nrc,1,self.dT))
        arand = trt.Weights(arand.astype(np.float32)*self.hps['model']['sigma'])
        a = self.nw.add_constant((N,self.nrc,1,self.dT), arand);a.name = "audio"

        for k in reversed(range(self.hps['model']['n_flows'])):
            nh = self.nrc // 2

            a0 = self.nw.add_slice(a.get_output(0), (0,0,0,0), (N,nh,1,self.dT), (1,1,1,1))
            a0.name = "slice_a0_in_flow_" + str(k)
            o  = self.wavenet(a0, c, k)

            log_s = self.nw.add_slice(o.get_output(0), (0,nh,0,0), (N,nh,1,self.dT), (1,1,1,1))
            log_s.name = "slice_log_s_in_flow_" + str(k)

            s  = self.nw.add_unary(log_s.get_output(0), trt.UnaryOperation.EXP)
            s.name = "log_s_to_s_by_exp_flow_" + str(k)
            b  = self.nw.add_slice(o.get_output(0), (0,0,0,0), (N,nh,1,self.dT), (1,1,1,1))
            b.name = "slice_b_in_flow_" + str(k)
            a1 = self.nw.add_slice(a.get_output(0), (0,nh,0,0), (N,nh,1,self.dT), (1,1,1,1))
            a1.name = "slice_a1_in_flow_" + str(k)
            a1 = self.nw.add_elementwise(a1.get_output(0), b.get_output(0), trt.ElementWiseOperation.SUB)
            a1.name = "a1_sub_b_in_flow_" + str(k)
            a1 = self.nw.add_elementwise(a1.get_output(0), s.get_output(0), trt.ElementWiseOperation.DIV)
            a1.name = "a1_div_s_in_flow_" + str(k)
            # a0 = self.nw.add_slice(a.get_output(0), (0,0,0,0), (N,nh,1,self.dT), (1,1,1,1))
            # a0.name = "slice_a0_again_in_flow_" + str(k) # WAR in TRT 5.1.5
            a  = self.nw.add_concatenation([a0.get_output(0), a1.get_output(0)])
            a.name = "concat_a0_a1_in_flow_" + str(k)
            W  = self.get_t('convinv.{}.conv.weight'.format(k), reverse=True)
            a  = self.nw.add_convolution(a.get_output(0), self.nrc, (1, 1), W, trt.Weights())
            a.name = "1X1Conv_in_flow_" + str(k)
            if k % self.hps['model']['n_early_every'] == 0 and k > 0:
                zrand = np.random.normal(0., 1., (N,self.nes,1,self.dT))
                zrand = trt.Weights(zrand.astype(np.float32) * self.hps['model']['sigma'])
                z = self.nw.add_constant((N,self.nes,1,self.dT), zrand)
                z.name = "audio_z_in_flow_" + str(k)
                a = self.nw.add_concatenation([z.get_output(0), a.get_output(0)])
                a.name = "z_concat_a_in_flow_" + str(k)
                a = self.nw.add_identity(a.get_output(0)) # WAR in TRT 5.1.5
                a.name = "identity_to_avoid_bug_in_flow_" + str(k)
                self.nrc = self.nrc + self.nes

        a = self.nw.add_shuffle(a.get_output(0))
        a.name = "final_shuffle_to_generate_audio_in_batch"
        a.first_transpose = (0,3,2,1); a.reshape_dims = (-1, self.dT*8)
        self.nw.mark_output(a.get_output(0))


    def wavenet(self, a, c, fid):
        """
        WaveNet like submodule, stacked to form WaveGlow
        Input:
            audio:       [batch_dim, half_audio_channel_dim, 1, time_dim]
            conditioner: [batch_dim, conditioner_channel_dim, 1, time_dim]
        Output:
            audio:       [batch_dim, audio_channel_dim, 1, time_dim]
        """
        global N

        ss = self.hps['model']['skip_size']
        rs = self.hps['model']['residue_size']

        W = self.get_t('WN.{}.start.weight'.format(fid))
        B = self.get_t('WN.{}.start.bias'.format(fid))
        a = self.nw.add_convolution(a.get_output(0), B.size, (1,1), W, B)
        a.name = "wn_start_conv_in_flow_" + str(fid)

        for idx in range(self.hps['model']['dilation_num']):
            conv = self.non_causal_dilated_conv(a, c, fid, idx)
            if idx < self.hps['model']['dilation_num'] - 1:
              rss = rs + ss # res_and_skip_size
            else:
              rss = ss

            W = self.get_t('WN.{}.res_skip_layers.{}.weight'.format(fid, idx))
            B = self.get_t('WN.{}.res_skip_layers.{}.bias'.format(fid, idx))
            conv = self.nw.add_convolution(conv.get_output(0), rss, (1,1), W, B)
            conv.name = "wn_res_skip_conv_in_flow_{}_layer_{}".format(fid, idx)
            if idx < self.hps['model']['dilation_num'] - 1:
                tmp = self.nw.add_slice(conv.get_output(0),(0,0,0,0),(N,ss,1,self.dT),(1,1,1,1))
                tmp.name = "wn_{}_dilation_slice1_in_flow_{}".format(idx, fid)
                a = self.nw.add_elementwise(a.get_output(0), tmp.get_output(0), trt.ElementWiseOperation.SUM)
                a.name = "wn_{}_dilation_sum_in_flow_{}".format(idx, fid)
                skip = self.nw.add_slice(conv.get_output(0),(0,rs,0,0),(N,ss,1,self.dT),(1,1,1,1))
                skip.name = "wn_{}_dilation_slice2_in_flow_{}".format(idx, fid)
            else:
                skip = conv

            if idx == 0:
                o = skip # o stands for output
            else:
                o = self.nw.add_elementwise(o.get_output(0),skip.get_output(0),trt.ElementWiseOperation.SUM)
                o.name = "wn_{}_dilation_output_sum_in_flow_{}".format(idx, fid)

        W = self.get_t('WN.{}.end.weight'.format(fid))
        B = self.get_t('WN.{}.end.bias'.format(fid))
        o = self.nw.add_convolution(o.get_output(0), B.size, (1,1), W, B)
        o.name = "wn_{}_end_conv_in_flow_" + str(fid)
        return o


    def non_causal_dilated_conv(self, a, c, fid, lid):
        '''
        layers to form WaveNet, gradually increase "receptive field" in time dim
        '''
        global N; ksz = self.hps['model']['kernel_size']

        W = self.get_t('WN.{}.in_layers.{}.weight'.format(fid, lid))
        B = self.get_t('WN.{}.in_layers.{}.bias'.format(fid, lid))

        conv = self.nw.add_convolution(a.get_output(0), B.size, (1,ksz), W, B)
        conv.dilation = (1,2**lid)
        conv.padding = (0, int((2**lid) * (ksz - 1) / 2)) # shift
        conv.name = "{}_dilated_conv_for_audio_in_flow_{}".format(lid, fid)

        W = self.get_t('WN.{}.cond_layers.{}.weight'.format(fid, lid))
        B = self.get_t('WN.{}.cond_layers.{}.bias'.format(fid, lid))
        cond = self.nw.add_convolution(c.get_output(0), B.size, (1,1), W, B)
        cond.name = "{}_conv_for_conditoner_in_flow_{}".format(lid, fid)

        t = self.nw.add_elementwise(cond.get_output(0), conv.get_output(0),
                                                trt.ElementWiseOperation.SUM)
        t.name = "audio_add_cond_in_flow_{}_conv_{}".format(fid, lid)

        ch = int(B.size / 2) # ch/f/s/o: channel, first and second half, output
        f = self.nw.add_slice(t.get_output(0), (0,0,0,0), (N,ch,1,self.dT), (1,1,1,1))
        f.name = "slice1_in_flow_{}_conv_{}".format(fid, lid)
        f = self.nw.add_activation(f.get_output(0), trt.ActivationType.TANH)
        f.name = "tanh_in_flow_{}_conv_{}".format(fid, lid)
        s = self.nw.add_slice(t.get_output(0), (0,ch,0,0), (N,ch,1,self.dT), (1,1,1,1))
        s.name = "slice2_in_flow_{}_conv_{}".format(fid, lid)
        s = self.nw.add_activation(s.get_output(0), trt.ActivationType.SIGMOID)
        s.name = "sigmoid_in_flow_{}_conv_{}".format(fid, lid)

        prod = self.nw.add_elementwise(f.get_output(0), s.get_output(0),
                                                trt.ElementWiseOperation.PROD)
        prod.name = "f_s_prod_in_flow_{}_conv_{}".format(fid, lid)

        return prod

if __name__ == '__main__':
    wg = WaveGlow()
    if len(sys.argv) < 2: # only scipt file name
        wg.build_graph()
        wg.build_engine()
    else: # load pre-built engine for infer
        wg.infer(sys.argv[1])
    wg.destory()
