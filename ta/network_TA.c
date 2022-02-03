#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "darknet_TA.h"
#include "blas_TA.h"
#include "network_TA.h"
#include "math_TA.h"

#include "darknetp_ta.h"
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

network_TA netta;
int roundnum = 0;
int layernum = 0;
float err_sum = 0;
float avg_loss = -1;

float *ta_net_input;
float *ta_net_delta;
float *ta_net_output;

void make_network_TA(int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches)
{
    netta.n = n;

    //netta.seen = calloc(1, sizeof(size_t));
    netta.seen = calloc(1, sizeof(uint64_t));
    //netta.layers = calloc(netta.n, sizeof(layer_TA));
    netta.layers = calloc(netta.n, sizeof(layer_TA *));
    netta.t    = calloc(1, sizeof(int));
    netta.cost = calloc(1, sizeof(float));

    netta.learning_rate = learning_rate;
    netta.momentum = momentum;
    netta.decay = decay;
    netta.time_steps = time_steps;
    netta.notruth = notruth;
    netta.batch = batch;
    netta.subdivisions = subdivisions;
    netta.random = random;
    netta.adam = adam;
    netta.B1 = B1;
    netta.B2 = B2;
    netta.eps = eps;
    netta.h = h;
    netta.w = w;
    netta.c = c;
    netta.inputs = inputs;
    netta.max_crop = max_crop;
    netta.min_crop = min_crop;
    netta.max_ratio = max_ratio;
    netta.min_ratio = min_ratio;
    netta.center = center;
    netta.clip = clip;
    netta.angle = angle;
    netta.aspect = aspect;
    netta.saturation = saturation;
    netta.exposure = exposure;
    netta.hue = hue;
    netta.burn_in = burn_in;
    netta.power = power;
    netta.max_batches = max_batches;
    netta.workspace_size = 0;

    //netta.truth = net->truth; ////// ing network.c train_network
}

void forward_network_TA()
{

    if(roundnum == 0){
        // ta_net_input malloc so not destroy before addition backward
        ta_net_input = malloc(sizeof(float) * netta.layers[0]->inputs * netta.layers[0]->batch);
        ta_net_delta = malloc(sizeof(float) * netta.layers[0]->inputs * netta.layers[0]->batch);

        if(netta.workspace_size){
            printf("workspace_size=%ld\n", netta.workspace_size);
            netta.workspace = calloc(1, netta.workspace_size);
        }
    }
    roundnum++;
  
    //////
    TEE_Time t;
    TEE_GetSystemTime(&t);
    if(layernum > 0)
        free_layer_TA(netta.layers[netta.index]);

    netta.index = layernum;

    printf("LayerNum : %d\n", layernum);
    layer_TA l = *(netta.layers[layernum]);

    if(l.delta){
        fill_cpu_TA(l.outputs * l.batch, 0, l.delta, 1);
    }

    l.forward_TA(l, netta);

    if(debug_summary_pass == 1){
        summary_array("forward_network / l.output", l.output, l.outputs*netta.batch);
    }

    netta.input = l.output;

    if(l.truth) {
        netta.truth = l.output;
    }
    //output of the network (for predict)
    // &&
    if(!netta.train && l.type == SOFTMAX_TA){
        ta_net_output = malloc(sizeof(float)*l.outputs*1);
        for(int z=0; z<l.outputs*1; z++){
            ta_net_output[z] = l.output[z];
        }
    }

    TEE_Time t2;
    TEE_GetSystemTime(&t2);
    printf("Layer[%d] : \n\
                Start : %ld sec, %ld mil\n\
                End   : %ld sec, %ld mil\n", layernum, t.seconds, t.millis, t2.seconds, t2.millis);
    layernum++;

    if(layernum == netta.n){    
        calc_network_cost_TA();
    }


    /////////////////////////////////////////////////////
/*
    if(roundnum == 0){
        // ta_net_input malloc so not destroy before addition backward
        ta_net_input = malloc(sizeof(float) * netta.layers[0].inputs * netta.layers[0].batch);
        ta_net_delta = malloc(sizeof(float) * netta.layers[0].inputs * netta.layers[0].batch);

        if(netta.workspace_size){
            printf("workspace_size=%ld\n", netta.workspace_size);
            netta.workspace = calloc(1, netta.workspace_size);
        }
    }

    roundnum++;
    int i;
    for(i = 0; i < netta.n; ++i){
//        printf("forward_network_TA: layer num: %d\n", i);
        netta.index = i;
        layer_TA l = netta.layers[i];

        if(l.delta){
            fill_cpu_TA(l.outputs * l.batch, 0, l.delta, 1);
        }

        l.forward_TA(l, netta);

        if(debug_summary_pass == 1){
            summary_array("forward_network / l.output", l.output, l.outputs*netta.batch);
        }

        netta.input = l.output;

        if(l.truth) {
            netta.truth = l.output;
        }
        //output of the network (for predict)
        // &&
        if(!netta.train && l.type == SOFTMAX_TA){
            ta_net_output = malloc(sizeof(float)*l.outputs*1);
            for(int z=0; z<l.outputs*1; z++){
                ta_net_output[z] = l.output[z];
            }
        }

        // if(i == netta.n - 1)  // ready to back REE for the rest forward pass
        // {
        //     ta_net_input = malloc(sizeof(float)*l.outputs*l.batch);
        //     for(int z=0; z<l.outputs*l.batch; z++){
        //         ta_net_input[z] = netta.input[z];
        //     }
        // }
    }

    calc_network_cost_TA();
*/
}


void update_network_TA(update_args_TA a)
{
    int i;
    for(i = 0; i < netta.n; ++i){
        layer_TA l = *(netta.layers[i]);
        if(l.update_TA){
            l.update_TA(l, a);
        }
    }
}


void calc_network_cost_TA()
{
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < netta.n; ++i){
        if(netta.layers[i]->cost){
            sum += netta.layers[i]->cost[0];
            ++count;
        }
    }
    *netta.cost = sum/count;
    err_sum += *netta.cost;
}


void calc_network_loss_TA(int n, int batch)
{
    float loss = (float)err_sum/(n*batch);

    if(avg_loss == -1) avg_loss = loss;
    avg_loss = avg_loss*.9 + loss*.1;

    char loss_char[20];
    char avg_loss_char[20];
    ftoa(loss, loss_char, 5);
    ftoa(avg_loss, avg_loss_char, 5);
    IMSG("loss = %s, avg loss = %s from the TA\n",loss_char, avg_loss_char);
    err_sum = 0;
}



//void backward_network_TA(float *ca_net_input, float *ca_net_delta)
void backward_network_TA(float *ca_net_input)
{
    int i;

    for(i = netta.n-1; i >= 0; --i){
        layer_TA l = *(netta.layers[i]);

        if(l.stopbackward) break;
        if(i == 0){
            for(int z=0; z<l.inputs*l.batch; z++){
             // note: both ca_net_input and ca_net_delta are pointer
                ta_net_input[z] = ca_net_input[z];
                //ta_net_delta[z] = ca_net_delta[z]; zeros removing
                ta_net_delta[z] = 0.0f;
            }

            netta.input = ta_net_input;
            netta.delta = ta_net_delta;
        }else{
            layer_TA prev = *(netta.layers[i-1]);
            netta.input = prev.output;
            netta.delta = prev.delta;
        }

        netta.index = i;
        l.backward_TA(l, netta);

        // when the first layer in TEE is a Dropout layer
        if((l.type == DROPOUT_TA) && (i == 0)){
            for(int z=0; z<l.inputs*l.batch; z++){
                ta_net_input[z] = l.output[z];
                ta_net_delta[z] = l.delta[z];
            }
            //netta.input = l.output;
            //netta.delta = l.delta;
        }
    }
}

void free_layer_TA(layer_TA *l)
{
    free_layer_member_TA(l);
    
    if(l) {
        free(l);
        l = NULL;
    }


    return ;
}

void free_layer_member_TA(layer_TA *l)
{
    if(l->type == DROPOUT_TA){
        if(l->rand)           free(l->rand);
#ifdef GPU
        if(l->rand_gpu)             cuda_free(l->rand_gpu);
#endif
        return;
    }
    if(l->cweights){
        free(l->cweights);
        l->cweights = NULL;
    }
    if(l->indexes){
        free(l->indexes);  
        l->indexes = NULL;
    }
    if(l->input_layers){       
        free(l->input_layers);
        l->input_layers = NULL;
    }
    if(l->input_sizes){
        free(l->input_sizes);
        l->input_sizes = NULL;
    }
    if(l->map){                
        free(l->map);
        l->map = NULL;
    }
    if(l->rand) {
        free(l->rand);
        l->rand = NULL;
    }
    if(l->cost)               free(l->cost);
    if(l->state)              free(l->state);
    if(l->prev_state)         free(l->prev_state);
    if(l->forgot_state)       free(l->forgot_state);
    if(l->forgot_delta)       free(l->forgot_delta);
    if(l->state_delta)        free(l->state_delta);
    if(l->concat)             free(l->concat);
    if(l->concat_delta)       free(l->concat_delta);
    if(l->binary_weights)     free(l->binary_weights);
    if(l->biases)             free(l->biases);
    if(l->bias_updates)       free(l->bias_updates);
    if(l->scales)             free(l->scales);
    if(l->scale_updates)      free(l->scale_updates);
    if(l->weights)            free(l->weights);
    if(l->weight_updates)     free(l->weight_updates);
    if(l->delta)              free(l->delta);
    if(l->output){
        //free(l->output);
        l->output = NULL;
    }
    if(l->squared)            free(l->squared);
    if(l->norms)              free(l->norms);
    if(l->spatial_mean)       free(l->spatial_mean);
    if(l->mean)               free(l->mean);
    if(l->variance)           free(l->variance);
    if(l->mean_delta)         free(l->mean_delta);
    if(l->variance_delta)     free(l->variance_delta);
    if(l->rolling_mean)       free(l->rolling_mean);
    if(l->rolling_variance)   free(l->rolling_variance);
    if(l->x)                  free(l->x);
    if(l->x_norm)             free(l->x_norm);
    if(l->m)                  free(l->m);
    if(l->v)                  free(l->v);
    if(l->z_cpu)              free(l->z_cpu);
    if(l->r_cpu)              free(l->r_cpu);
    if(l->h_cpu)              free(l->h_cpu);
    if(l->binary_input)       free(l->binary_input);

#ifdef GPU
    if(l->indexes_gpu)           cuda_free((float *)l->indexes_gpu);

    if(l->z_gpu)                   cuda_free(l->z_gpu);
    if(l->r_gpu)                   cuda_free(l->r_gpu);
    if(l->h_gpu)                   cuda_free(l->h_gpu);
    if(l->m_gpu)                   cuda_free(l->m_gpu);
    if(l->v_gpu)                   cuda_free(l->v_gpu);
    if(l->prev_state_gpu)          cuda_free(l->prev_state_gpu);
    if(l->forgot_state_gpu)        cuda_free(l->forgot_state_gpu);
    if(l->forgot_delta_gpu)        cuda_free(l->forgot_delta_gpu);
    if(l->state_gpu)               cuda_free(l->state_gpu);
    if(l->state_delta_gpu)         cuda_free(l->state_delta_gpu);
    if(l->gate_gpu)                cuda_free(l->gate_gpu);
    if(l->gate_delta_gpu)          cuda_free(l->gate_delta_gpu);
    if(l->save_gpu)                cuda_free(l->save_gpu);
    if(l->save_delta_gpu)          cuda_free(l->save_delta_gpu);
    if(l->concat_gpu)              cuda_free(l->concat_gpu);
    if(l->concat_delta_gpu)        cuda_free(l->concat_delta_gpu);
    if(l->binary_input_gpu)        cuda_free(l->binary_input_gpu);
    if(l->binary_weights_gpu)      cuda_free(l->binary_weights_gpu);
    if(l->mean_gpu)                cuda_free(l->mean_gpu);
    if(l->variance_gpu)            cuda_free(l->variance_gpu);
    if(l->rolling_mean_gpu)        cuda_free(l->rolling_mean_gpu);
    if(l->rolling_variance_gpu)    cuda_free(l->rolling_variance_gpu);
    if(l->variance_delta_gpu)      cuda_free(l->variance_delta_gpu);
    if(l->mean_delta_gpu)          cuda_free(l->mean_delta_gpu);
    if(l->x_gpu)                   cuda_free(l->x_gpu);
    if(l->x_norm_gpu)              cuda_free(l->x_norm_gpu);
    if(l->weights_gpu)             cuda_free(l->weights_gpu);
    if(l->weight_updates_gpu)      cuda_free(l->weight_updates_gpu);
    if(l->biases_gpu)              cuda_free(l->biases_gpu);
    if(l->bias_updates_gpu)        cuda_free(l->bias_updates_gpu);
    if(l->scales_gpu)              cuda_free(l->scales_gpu);
    if(l->scale_updates_gpu)       cuda_free(l->scale_updates_gpu);
    if(l->output_gpu)              cuda_free(l->output_gpu);
    if(l->delta_gpu)               cuda_free(l->delta_gpu);
    if(l->rand_gpu)                cuda_free(l->rand_gpu);
    if(l->squared_gpu)             cuda_free(l->squared_gpu);
    if(l->norms_gpu)               cuda_free(l->norms_gpu);
#endif
}
