#ifndef LAYERS_TA_H
#define LAYERS_TA_H

#include "darknet_TA.h"
#include "avgpool_layer_TA.h"
#include "batchnorm_layer_TA.h"
#include "connected_layer_TA.h"
#include "convolutional_layer_TA.h"
#include "cost_layer_TA.h"
#include "dropout_layer_TA.h"
#include "maxpool_layer_TA.h"
#include "softmax_layer_TA.h"

union param{
	int i;
	float f;
	ACTIVATION_TA A;
	COST_TYPE_TA C;
};

typedef struct Param_ST{
	LAYER_TYPE_TA type;
	int batch;
	union param p[16];
}Param_ST;

typedef struct Param_WI {
    float *vec;
    int length;
    int layer_i;
    int additional;
    char type;
} Param_WI;

extern Param_ST *layer_param;
extern Param_WI *weight_param;

#endif

