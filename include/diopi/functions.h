/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/

#ifndef _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_H_
#define _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_H_

#include <diopi/diopirt.h>

#if defined(__cplusplus)
extern "C" {
#endif // __cplusplus


typedef enum {
    ReductionNone,
    ReductionMean,
    ReductionSum,
    ReductionEND
} diopiReduction_t;

typedef enum {
    RoundModeNone,
    RoundModeTrunc,
    RoundModeFloor,
    RoundModeEND
} diopiRoundMode_t;

typedef struct {
    diopiDtype_t stype;
    union {
        double  fval;
        int64_t ival;
    };
} diopiScalar_t;


/**
 * \brief get the vendor's name who implements the functions
 */
DIOPI_API const char* diopiGetVendorName();
DIOPI_API const char* diopiGetImplVersion();

/**
 * \brief Applies a 2D convolution over an input image composed of several input planes.
 */
DIOPI_API diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
                                          const diopiTensorHandle_t weight, const diopiTensorHandle_t bias, diopiSize_t stride,
                                          diopiSize_t padding, diopiSize_t dilation, int64_t groups);

DIOPI_API diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                  diopiTensorHandle_t grad3, const diopiTensorHandle_t grad_output, const diopiTensorHandle_t input,
                                                  const diopiTensorHandle_t weight, diopiSize_t *bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                  diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups);

/**
 * \brief Applies Batch Normalization for each channel across a batch of data.
 */
DIOPI_API diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean,
                                      diopiTensorHandle_t save_invstd, const diopiTensorHandle_t input, const diopiTensorHandle_t weight,
                                      const diopiTensorHandle_t bias, const diopiTensorHandle_t running_mean,
                                      const diopiTensorHandle_t running_var, bool training, double momentum, double eps);
DIOPI_API diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                              diopiTensorHandle_t grad_bias, const diopiTensorHandle_t grad_output, const diopiTensorHandle_t input, const diopiTensorHandle_t weight,
                                              const diopiTensorHandle_t running_mean, const diopiTensorHandle_t running_var, const diopiTensorHandle_t save_mean, 
                                              const diopiTensorHandle_t save_invstd, bool training, double eps);

/**
 * \brief Applies the rectified linear unit function element-wise
 */
DIOPI_API diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

#if defined(__cplusplus)
}
#endif // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_H_
