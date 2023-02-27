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
#endif  // __cplusplus


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
DIOPI_RT_API const char* diopiGetVendorName();
DIOPI_RT_API const char* diopiGetImplVersion();
DIOPI_RT_API const char* diopiGetLastErrorString();

/**
 * \brief Applies a 2D convolution over an input image composed of several input planes.
 */
DIOPI_API diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride,
                                          diopiSize_t padding, diopiSize_t dilation, int64_t groups);

DIOPI_API diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                  diopiTensorHandle_t grad3, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                  diopiConstTensorHandle_t weight, diopiSize_t *bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                  diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups);

/**
 * \brief Applies Batch Normalization for each channel across a batch of data.
 */
DIOPI_API diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean,
                                      diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                      diopiConstTensorHandle_t bias, diopiTensorHandle_t running_mean,
                                      diopiTensorHandle_t running_var, bool training, double momentum, double eps);
DIOPI_API diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                              diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                              diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var, diopiConstTensorHandle_t save_mean,
                                              diopiConstTensorHandle_t save_invstd, bool training, double eps);

/**
 * \brief Applies the rectified linear unit function element-wise
 */
DIOPI_API diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
DIOPI_API diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

DIOPI_API diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                     const diopiScalar_t* min_val, const diopiScalar_t* max_val);
DIOPI_API diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val);
DIOPI_API diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val);

DIOPI_API diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                     const diopiScalar_t* threshold, const diopiScalar_t* value);
DIOPI_API diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value);
DIOPI_API diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t input, const diopiScalar_t* threshold);

/**
 * \brief Applies the gaussian error linear unit function element-wise
 */
DIOPI_API diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                 diopiConstTensorHandle_t input, const char* approximate);
DIOPI_API diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                         diopiConstTensorHandle_t input, const char* approximate);

/**
 * \brief Applies element-wise, LeakyReLU(x) = max(0,x) + negative_slope*min(0,x)
 */
DIOPI_API diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                      diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope);
DIOPI_API diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* negative_slope);
DIOPI_API diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope, bool input_is_result);

/**
 * \brief Applies 2D average-pooling operation in kH×kW regions by step size sH×sW steps.
 */
DIOPI_API diopiError_t diopiAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode,
                                      bool count_include_pad, const int64_t* divisor_override);
DIOPI_API diopiError_t diopiAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                              diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                              diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode,
                                              bool count_include_pad, const int64_t* divisor_override);

/**
 * \brief Applies a 2D max pooling over an input signal composed of several input planes
 */
DIOPI_API diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode);
DIOPI_API diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices,
                                                 diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride,
                                                 diopiSize_t padding, diopiSize_t dilation, bool ceil_mode);
DIOPI_API diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                              diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices);

/**
 * \brief Applies a 2D adaptive average pooling over an input signal composed of several input planes.
 */
DIOPI_API diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                              diopiConstTensorHandle_t input, diopiSize_t output_size);
DIOPI_API diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                                      diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input);

/**
 * \brief Applies a 2D adaptive max pooling over an input signal composed of several input planes.
 */
DIOPI_API diopiError_t diopiAdaptiveMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                              diopiConstTensorHandle_t input, diopiSize_t output_size);
DIOPI_API diopiError_t diopiAdaptiveMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices,
                                                         diopiConstTensorHandle_t input, diopiSize_t output_size);
DIOPI_API diopiError_t diopiAdaptiveMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices);

/**
 * \brief Randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
 */
DIOPI_API diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask,
                                    diopiConstTensorHandle_t input, double p, bool train);
DIOPI_API diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask,
                                       double p, bool train);

/**
 * \brief Measures the element-wise mean squared error
 */
DIOPI_API diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t target, diopiReduction_t reduction);
DIOPI_API diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiSigmoidFocalLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t inputs,
                                             diopiConstTensorHandle_t targets, float alpha, float gamma, diopiReduction_t reduction);
DIOPI_API diopiError_t diopiSigmoidFocalLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output,
                                                     diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                                     diopiTensorHandle_t grad_input, float gamma, float alpha, diopiReduction_t reduction);

/**
 * \brief Measures thee Cross Entropy between the target and input probabilities.
 */
DIOPI_API diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                             diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction,
                                             int64_t ignore_index, double label_smoothing);
DIOPI_API diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                     diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                     diopiReduction_t reduction, int64_t ignore_index, double label_smoothing);

/**
 * \brief Measures thee nll loss between the target and input probabilities.
 */
DIOPI_API diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction,
                                    int64_t ignore_index);
DIOPI_API diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                            diopiReduction_t reduction, int64_t ignore_index);

/**
 * \brief Measures the Binary Cross Entropy between the target and input probabilities.
 */
DIOPI_API diopiError_t diopiBCEWithLogits(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                          diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction);
DIOPI_API diopiError_t diopiBCEWithLogitsBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                  diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction);
DIOPI_API diopiError_t diopiBCELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                    diopiConstTensorHandle_t weight, diopiReduction_t reduction);
DIOPI_API diopiError_t diopiBCELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction);

/**
 * \brief Element-wise math functions
 */
DIOPI_API diopiError_t diopiSign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiAbsInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiAbs(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiNegInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiNeg(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiFloorInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiFloor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiSinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiSin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiTanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
DIOPI_API diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                         diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output);

DIOPI_API diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
DIOPI_API diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                            diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output);

DIOPI_API diopiError_t diopiExpInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiExp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiLog2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiLog10(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiErfInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiErf(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent);
DIOPI_API diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent);
DIOPI_API diopiError_t diopiPowInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* exponent);
DIOPI_API diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent);
DIOPI_API diopiError_t diopiPowInpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t exponent);

DIOPI_API diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                diopiConstTensorHandle_t other, const diopiScalar_t* alpha);
DIOPI_API diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
                                   diopiConstTensorHandle_t other, const diopiScalar_t* alpha);
DIOPI_API diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      const diopiScalar_t* other, const diopiScalar_t* alpha);
DIOPI_API diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
                                         const diopiScalar_t* other, const diopiScalar_t* alpha);

DIOPI_API diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                diopiConstTensorHandle_t other, const diopiScalar_t* alpha);

DIOPI_API diopiError_t diopiSubInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
                                   diopiConstTensorHandle_t other, const diopiScalar_t* alpha);

DIOPI_API diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      const diopiScalar_t* other, const diopiScalar_t* alpha);

DIOPI_API diopiError_t diopiSubInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
                                         const diopiScalar_t* other, const diopiScalar_t* alpha);

DIOPI_API diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

DIOPI_API diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode);
DIOPI_API diopiError_t diopiDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
                                diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode);
DIOPI_API diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      const diopiScalar_t* other, diopiRoundMode_t rounding_mode);
DIOPI_API diopiError_t diopiDivInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
                                         const diopiScalar_t* other, diopiRoundMode_t rounding_mode);

/**
 * \brief Broadcast-BLAS functions
 */
DIOPI_API diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2);

DIOPI_API diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value);

DIOPI_API diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                   diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value);

DIOPI_API diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t mat1, diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha);

DIOPI_API diopiError_t diopiCholesky(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t info, diopiConstTensorHandle_t mat, bool upper, bool checkerror);
DIOPI_API diopiError_t diopiCholeskyBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_mat, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t L, bool upper);

DIOPI_API diopiError_t diopiTriangularSolve(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t cloned_mat, diopiConstTensorHandle_t b,
                                            diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular);
DIOPI_API diopiError_t diopiTriangularSolveBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_b, diopiTensorHandle_t grad_mat, diopiConstTensorHandle_t grad_x, diopiConstTensorHandle_t grad_cloned_mat,
                                                    diopiConstTensorHandle_t x, diopiConstTensorHandle_t b, diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular);

/**
 * \brief Clamps all elements in input into the range [ min, max ].
 */
DIOPI_API diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max);
DIOPI_API diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max);
DIOPI_API diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max);
DIOPI_API diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t min, diopiConstTensorHandle_t max);

DIOPI_API diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max);
DIOPI_API diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max);
DIOPI_API diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max);
DIOPI_API diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max);
DIOPI_API diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min);
DIOPI_API diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min);
DIOPI_API diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min);
DIOPI_API diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min);

/**
 * \brief Fills elements of self tensor with value.
 */
DIOPI_API diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value);

/**
 * \brief Computes the element-wise logical AND/OR of the given input tensors.
 */
DIOPI_API diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiBitwiseAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiBitwiseAndInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiBitwiseOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiBitwiseOrInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

/**
 * \brief Computes element-wise comparison, including =, !=, >=, >, <= and <.
 */

DIOPI_API diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiEqInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiEqInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiNeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiNeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiGeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiGeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiGtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiGtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiLeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiLeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiLtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiLtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * \brief Returns the mean value of all elements in the input tensor.
 */
DIOPI_API diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                 diopiConstTensorHandle_t input, diopiSize_t dim, diopiDtype_t dtype);

/**
 * \brief Returns the sum value of all elements in the input tensor.
 */
DIOPI_API diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                 diopiConstTensorHandle_t input, diopiSize_t dim, diopiDtype_t dtype);

/**
 * \brief Returns the standard derivation of all elements in the input tensor.
 */
DIOPI_API diopiError_t diopiStd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                diopiConstTensorHandle_t input, diopiSize_t dim, bool unbiased);

/**
 * \brief Returns the minimum value of all elements in the input tensor.
 */
DIOPI_API diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t min_indices,
                                diopiConstTensorHandle_t input, int64_t dim);
DIOPI_API diopiError_t diopiMinAll(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiConstTensorHandle_t input);

/**
 * \brief Returns the maximum value of all elements in the input tensor.
 */
DIOPI_API diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices,
                                diopiConstTensorHandle_t input, int64_t dim);
DIOPI_API diopiError_t diopiMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input);

/**
 * \brief Returns True if any element in each row of the tensor in the given dimension dim are True, False otherwise.
 */
DIOPI_API diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t* dim);

/**
 * \brief Returns True if all elements in each row of the tensor in the given dimension dim are True, False otherwise.
 */
DIOPI_API diopiError_t diopiAll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t* dim);

/**
 * \brief Applies a softmax function.
 */
DIOPI_API diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiDtype_t dtype);
DIOPI_API diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t output, int64_t dim, diopiDtype_t input_dtype);

/**
 * \brief Applies a log_softmax function.
 */
DIOPI_API diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiDtype_t dtype);
DIOPI_API diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                               diopiConstTensorHandle_t output, int64_t dim, diopiDtype_t input_dtype);

/**
 * \brief Returns a new tensor which indexes the input tensor along dimension dim using the entries in index.
 */
DIOPI_API diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t* indices, int64_t nums);
DIOPI_API diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t zeros_like_input,
                                          diopiConstTensorHandle_t* indices, int64_t nums, diopiConstTensorHandle_t grad);

DIOPI_API diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                        diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index);
DIOPI_API diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad,
                                                diopiSize_t input_sizes, int64_t dim, diopiConstTensorHandle_t index);

/**
 * \brief Slices the input tensor along the selected dimension at the given index.
 */
DIOPI_API diopiError_t diopiSelect(diopiContextHandle_t ctx,  diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t index);
DIOPI_API diopiError_t diopiSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                           diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t index);

/**
 * \brief Embeds the values of the src tensor into input at the given index/dimension. This function returns a tensor with fresh storage; it does not create a view.
 */
DIOPI_API diopiError_t diopiSelectScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t src, int64_t dim, int64_t index);
DIOPI_API diopiError_t diopiSliceScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                         diopiConstTensorHandle_t src, int64_t dim, int64_t start, int64_t end, int64_t step);
/**
 * \brief Slices the input tensor along the selected dimension at the given index.
 */
DIOPI_API diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t null_out, diopiConstTensorHandle_t input,
                                  int64_t dim, int64_t start, int64_t end, int64_t step);
DIOPI_API diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                          diopiSize_t input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step);

/**
 * \brief Copies elements from source into self tensor at positions where the mask is True.
 */
DIOPI_API diopiError_t diopiMaskedScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t mask, diopiConstTensorHandle_t source);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiNms(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t dets,
                                diopiConstTensorHandle_t scores, double iou_threshold);

/**
 * \brief Returns a tensor containing the indices of all non-zero elements of input.
 */
DIOPI_API diopiError_t diopiNonzero(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input);

/**
 * \brief Applies a linear transformation to the incoming data: y=xAT+b.
 */
DIOPI_API diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                   diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias);
DIOPI_API diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                           diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiRoiAlign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                     diopiConstTensorHandle_t rois, double spatial_scale, int64_t pooled_height,
                                     int64_t pooled_width, int64_t sampling_ratio, bool aligned);

DIOPI_API diopiError_t diopiRoiAlignBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad,
                                             diopiConstTensorHandle_t rois, double spatial_scale, int64_t pooled_height,
                                             int64_t pooled_width, int64_t batch_size, int64_t channels, int64_t height,
                                             int64_t width, int64_t sampling_ratio, bool aligned);

/**
 * \brief Implements stochastic gradient descent optimizer
 */
DIOPI_API diopiError_t diopiSgd(diopiContextHandle_t ctx, diopiTensorHandle_t w, diopiTensorHandle_t dw, diopiTensorHandle_t buf,
                                double lr, double momentum, double dampening, double weight_decay, bool nesterov);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiClipGradNorm(diopiContextHandle_t ctx, double* out, diopiTensorHandle_t *parameters,
                                         int64_t num_parameters, double max_norm, double norm_type, bool error_if_nonfinite);

/**
 * \brief A simple lookup table that looks up embeddings in a fixed dictionary and size.
 */
DIOPI_API diopiError_t diopiEmbeddingRenorm_(diopiContextHandle_t ctx, diopiTensorHandle_t inout,
                                             diopiConstTensorHandle_t indices, double max_norm, double norm_type);
DIOPI_API diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight,
                                      diopiConstTensorHandle_t indices, int64_t padding_idx, bool scale_grad_byfreq, bool sparse);
DIOPI_API diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad,
                                              diopiConstTensorHandle_t indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_byfreq, bool sparse);

/**
 * \brief Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input.
 */
DIOPI_API diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal);

/**
 * \brief Concatenates the given sequence of seq tensors in the given dimension.
 */
DIOPI_API diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t num_inputs, int64_t dim);

/**
 * \brief Splits the tensor into chunks.
 */
DIOPI_API diopiError_t diopiSplitWithSizes(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, int64_t num_outs,
                                           diopiConstTensorHandle_t input, const diopiSize_t splitSizes, int64_t dim);

/**
 * \brief Concatenates a sequence of tensors along a new dimension.
 */
DIOPI_API diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                  diopiConstTensorHandle_t* tensors, int64_t numTensors, int64_t dim);

/**
 * \brief Sorts the elements of the input tensor along a given dimension in ascending order by
 * value.
 */
DIOPI_API diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices,
                                 diopiConstTensorHandle_t input, int64_t dim, bool descending, const bool* stable);

/**
 * \brief Returns the k largest elements of the given input tensor along a given dimension.
 */
DIOPI_API diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices,
                                 diopiConstTensorHandle_t input, int64_t k, int64_t dim, bool largest, bool sorted);

/**
 * \brief Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1
 * are swapped.
 */
DIOPI_API diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                      diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1);

/**
 * \brief Returns a long tensor that has one more dimension with 1 values at the
 *        index of last dimension indicated by the input, and 0 everywhere else.
 */
DIOPI_API diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                   diopiConstTensorHandle_t input, int64_t num_classes);

/**
 * \brief Return a tensor of elements selected from either x or y, depending on condition.
 */
DIOPI_API diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition,
                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * \brief Fills elements of self tensor with value where mask is True.
 */
DIOPI_API diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                       diopiConstTensorHandle_t value);
DIOPI_API diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value);
DIOPI_API diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                             const diopiScalar_t* value);
DIOPI_API diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value);

/**
 * \brief Computes the reciprocal of the elements of input.
 */
DIOPI_API diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
DIOPI_API diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t input);

/**
 * \brief Implements AdamW optimizer.
 */
DIOPI_API diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad,
                                  diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq,
                                  float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad);

/**
 * \brief Applies a 2D transposed convolution operator over an input image composed of several input planes, sometimes also called “deconvolution”.
 */
DIOPI_API diopiError_t diopiConvTranspose2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride,
                                            diopiSize_t padding, diopiSize_t output_padding, int64_t groups, diopiSize_t dilation);

/**
 * \brief Extracts sliding local blocks from a batched input tensor.
 */
DIOPI_API diopiError_t diopiUnfold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t size, int64_t step);
DIOPI_API diopiError_t diopiUnfoldBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiSize_t input_sizes, int64_t dim, int64_t size, int64_t step);

/**
 * \brief Returns the cumulative sum of elements of input in the dimension dim.
 */
DIOPI_API diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiDtype_t dtype);

/**
 * \brief Computes batched the p-norm distance between each pair of the two collections of row vectors.
 */
DIOPI_API diopiError_t diopiCdist(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2,
                                  double p, const int64_t* compute_mode);
DIOPI_API diopiError_t diopiCdistBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                          diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p, diopiConstTensorHandle_t cdist);

/**
 * \brief Computes the element-wise logical Not of the given input tensors.
 */
DIOPI_API diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * \brief Returns the indices of the maximum values of a tensor across a dimension.
 */
DIOPI_API diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim);

/**
 * \brief Implements Adadelta algorithm.
 */
DIOPI_API diopiError_t diopiAdadelta(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg,
                                     diopiTensorHandle_t acc_delta, float lr, float rho, float eps, float weight_decay);

/**
 * \brief Implements Adam optimizer.
 */
DIOPI_API diopiError_t diopiAdam(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq,
                                 diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad);

/**
 * \brief Creates a criterion that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise.
 */
DIOPI_API diopiError_t diopiSmoothL1Loss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                         diopiReduction_t reduction, double beta);
DIOPI_API diopiError_t diopiSmoothL1LossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction, double beta);

/**
 * \brief Applies a 3D convolution over an input image composed of several input planes.
 */
DIOPI_API diopiError_t diopiConvolution3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride,
                                          diopiSize_t padding, diopiSize_t dilation, int64_t groups);
DIOPI_API diopiError_t diopiConvolution3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                  diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                  diopiConstTensorHandle_t weight, diopiSize_t *bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                  diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups);

/**
 * \brief Applies a 3D max pooling over an input signal composed of several input planes
 */
DIOPI_API diopiError_t diopiMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode);
DIOPI_API diopiError_t diopiMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices,
                                                 diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride,
                                                 diopiSize_t padding, diopiSize_t dilation, bool ceil_mode);
DIOPI_API diopiError_t diopiMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                              diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices);

/**
 * \brief Applies a 3D adaptive average pooling over an input signal composed of several input planes.
 */
DIOPI_API diopiError_t diopiAdaptiveAvgPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                              diopiConstTensorHandle_t input, diopiSize_t output_size);
DIOPI_API diopiError_t diopiAdaptiveAvgPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                                      diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input);

/**
 * \brief Applies a 3D adaptive max pooling over an input signal composed of several input planes.
 */
DIOPI_API diopiError_t diopiAdaptiveMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                              diopiConstTensorHandle_t input, diopiSize_t output_size);
DIOPI_API diopiError_t diopiAdaptiveMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices,
                                              diopiConstTensorHandle_t input, diopiSize_t output_size);
DIOPI_API diopiError_t diopiAdaptiveMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices);

/**
 * \brief Returns a new 1-D tensor which indexes the input tensor according to the boolean mask.
 */
DIOPI_API diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask);
DIOPI_API diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                                 diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask);

/**
 * \brief Element-wise math functions.
 */
DIOPI_API diopiError_t diopiMaximum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiMinimum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2);

/**
 * \brief Fills the elements of the input tensor with value by selecting the indices in the order given in index.
 */
DIOPI_API diopiError_t diopiIndexFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                            int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value);
DIOPI_API diopiError_t diopiIndexFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value);
DIOPI_API diopiError_t diopiIndexFillInpScalar(diopiContextHandle_t ctx, diopiConstTensorHandle_t input,
                                               int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value);
DIOPI_API diopiError_t diopiIndexFillInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t input,
                                         int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value);

/**
 * \brief Expand tensor to the same size as size.
 */
DIOPI_API diopiError_t diopiExpand(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size);

/**
 * \brief Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
 */
DIOPI_API diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps);

/**
 * \brief Returns a new tensor with its dimensions permuted.
 */
DIOPI_API diopiError_t diopiPermute(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims);

/**
 * \brief Pads tensor.
 */
DIOPI_API diopiError_t diopiPad(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char* mode, double* value);

/**
 * \brief Roll the tensor along the given dimension(s).
 */
DIOPI_API diopiError_t diopiRoll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t shifts, diopiSize_t dims);

/**
 * \brief Reverse the order of a n-D tensor along given axis in dims.
 */
DIOPI_API diopiError_t diopiFlip(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims);

/**
 * \brief Returns the matrix norm or vector norm of a given tensor.
 */
DIOPI_API diopiError_t diopiNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim, diopiDtype_t dtype);

/**
 * \brief Applies Group Normalization over a mini-batch of inputs.
 */
DIOPI_API diopiError_t diopiGroupNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, int64_t num_groups, double eps);
DIOPI_API diopiError_t diopiGroupNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                              diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t mean,
                                              diopiConstTensorHandle_t rstd, int64_t num_groups);

/**
 * \brief Returns the unique elements of the input tensor.
 */
DIOPI_API diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, int64_t* dim,
                                   bool sorted, bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts);

/**
 * \brief Returns the product of all elements in the input tensor.
 */
DIOPI_API diopiError_t diopiProd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t* dim, diopiDtype_t type);

/**
 * \brief Computes the Connectionist Temporal Classification loss.
 */
DIOPI_API diopiError_t diopiCTCLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t neg_log_likelihood, diopiTensorHandle_t log_alpha,
                                    diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths,
                                    diopiConstTensorHandle_t target_lengths, int64_t blank, diopiReduction_t reduction, bool zero_infinity);
DIOPI_API diopiError_t diopiCTCLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets,
                                            diopiConstTensorHandle_t input_lengths, diopiConstTensorHandle_t target_lengths, diopiConstTensorHandle_t neg_log_likelihood, diopiConstTensorHandle_t log_alpha,
                                            int64_t blank, diopiReduction_t reduction, bool zero_infinity);

/**
 * \brief Applies modulus operation.
 */
DIOPI_API diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiScalar_t* other);
DIOPI_API diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiScalar_t* input, diopiConstTensorHandle_t other);

/**
 * \brief Gathers values along an axis specified by dim.
 */
DIOPI_API diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index);
DIOPI_API diopiError_t diopiGatherBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index);

/**
 * \brief Writes all values from the tensor src into input at the indices specified in the index tensor.
 */
DIOPI_API diopiError_t diopiScatterInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce);
DIOPI_API diopiError_t diopiScatterInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce);
DIOPI_API diopiError_t diopiScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce);
DIOPI_API diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce);

/**
 * \brief Puts values from the tensor values into the tensor input using the indices specified in indices.
 */
DIOPI_API diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices, int64_t indices_counts, bool accumulate);
DIOPI_API diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices, int64_t indices_counts, bool accumulate);

/**
 * \brief Distribution and random numbers.
 */
DIOPI_API diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, int64_t idx);
DIOPI_API diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, int64_t idx);
DIOPI_API diopiError_t diopiBernoulli(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t idx);
DIOPI_API diopiError_t diopiBernoulliInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t idx);
DIOPI_API diopiError_t diopiBernoulliScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, int64_t idx);
DIOPI_API diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, const diopiScalar_t* step);
DIOPI_API diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, int64_t idx);

/**
 * \brief Applies Layer Normalization over a mini-batch of inputs.
 */
DIOPI_API diopiError_t diopiLayerNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                      diopiSize_t normalized_shape, double eps);
DIOPI_API diopiError_t diopiLayerNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                              diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                              diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, diopiSize_t normalized_shape);

/**
 * \brief Copies the elements from src into input tensor.
 */
DIOPI_API diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t input);

/**
 * \brief Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.
 */
DIOPI_API diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size);
DIOPI_API diopiError_t diopiUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                    diopiSize_t out_size, diopiSize_t in_size);
DIOPI_API diopiError_t diopiUpsampleLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size,
                                           bool align_corners, const char* mode);
DIOPI_API diopiError_t diopiUpsampleLinearBackward(diopiContextHandle_t ctx,  diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                   diopiSize_t out_size, diopiSize_t in_size, bool align_corners, const char* mode);

/**
 * \brief Computes the inverse error function of input tensor.
 */
DIOPI_API diopiError_t diopiErfinv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
DIOPI_API diopiError_t diopiErfinvInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * \brief Extracts sliding local blocks from a batched input tensor.
 */
DIOPI_API diopiError_t diopiIm2Col(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                   diopiSize_t kernel_size, diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride);

/**
 * \brief Combines an array of sliding local blocks into a large containing tensor.
 */
DIOPI_API diopiError_t diopiCol2Im(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                   diopiSize_t output_size, diopiSize_t kernel_size, diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride);


/* DIOPI functions from MMCV extension ops */

/**
 * \brief
 */
DIOPI_API diopiError_t diopiAssignScoreWithk(diopiContextHandle_t ctx, diopiConstTensorHandle_t points, diopiConstTensorHandle_t centers,
                                diopiConstTensorHandle_t scores, diopiConstTensorHandle_t knn_idx,
                                diopiTensorHandle_t output, int64_t B, int64_t N0, int64_t N1, int64_t M,
                                int64_t K, int64_t O, int64_t aggregate);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiAssignScoreWithkBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t points,
                                 diopiConstTensorHandle_t centers, diopiConstTensorHandle_t scores,
                                 diopiConstTensorHandle_t knn_idx, diopiTensorHandle_t grad_points,
                                 diopiTensorHandle_t grad_centers, diopiTensorHandle_t grad_scores,
                                 int64_t B, int64_t N0, int64_t N1, int64_t M, int64_t K, int64_t O,
                                 int64_t aggregate);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiBallQuery(diopiContextHandle_t ctx, diopiConstTensorHandle_t new_xyz, diopiConstTensorHandle_t xyz, diopiTensorHandle_t idx, int64_t b, int64_t n, int64_t m,
                                      float min_radius, float max_radius, int64_t nsample);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiStackBallQuery(diopiContextHandle_t ctx, diopiConstTensorHandle_t new_xyz, diopiConstTensorHandle_t new_xyz_batch_cnt, diopiConstTensorHandle_t xyz,
                                           diopiConstTensorHandle_t xyz_batch_cnt, diopiTensorHandle_t idx, float max_radius, int64_t nsample);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiBboxOverlaps(diopiContextHandle_t ctx, diopiConstTensorHandle_t bboxes1, diopiConstTensorHandle_t bboxes2, diopiTensorHandle_t ious,
                        int64_t mode, bool aligned, int64_t offset);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiBorderAlign(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t boxes,
                               diopiTensorHandle_t output, diopiTensorHandle_t argmax_idx,
                               int64_t pool_size);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiBorderAlignBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t boxes,
                                diopiConstTensorHandle_t argmax_idx, diopiTensorHandle_t grad_input,
                                int64_t pool_size);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiBoxIouRotated(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes1, diopiConstTensorHandle_t boxes2, diopiTensorHandle_t ious,
                                          int64_t mode_flag, bool aligned);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiBoxIouQuadri(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes1, diopiConstTensorHandle_t boxes2, diopiTensorHandle_t ious,
                                         int64_t mode_flag, bool aligned);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiCARAFE(diopiContextHandle_t ctx, diopiConstTensorHandle_t features, diopiConstTensorHandle_t masks, diopiTensorHandle_t rfeatures,
                                   diopiTensorHandle_t routput, diopiTensorHandle_t rmasks, diopiTensorHandle_t output, int64_t kernel_size, int64_t group_size, int64_t scale_factor);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiCARAFEBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t top_grad, diopiConstTensorHandle_t rfeatures, diopiConstTensorHandle_t masks,
                                           diopiTensorHandle_t rtop_grad, diopiTensorHandle_t rbottom_grad_hs, diopiTensorHandle_t rbottom_grad, diopiTensorHandle_t rmask_grad,
                                           diopiTensorHandle_t bottom_grad, diopiTensorHandle_t mask_grad, int64_t kernel_size, int64_t group_size, int64_t scale_factor);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiCARAFENAIVE(diopiContextHandle_t ctx, diopiConstTensorHandle_t features,
                                        diopiConstTensorHandle_t masks, diopiTensorHandle_t output,
                                        int64_t kernel_size,
                                        int64_t group_size,
                                        int64_t scale_factor);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiCARAFENAIVEBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t top_grad, diopiConstTensorHandle_t features, diopiConstTensorHandle_t masks,
                                        diopiTensorHandle_t bottom_grad, diopiTensorHandle_t mask_grad, int64_t kernel_size, int64_t group_size, int64_t scale_factor);


/**
 * \brief
 */
DIOPI_API diopiError_t diopiCorrelation(diopiContextHandle_t ctx, diopiTensorHandle_t input1, diopiTensorHandle_t input2, diopiTensorHandle_t output, int64_t kH,
                         int64_t kW, int64_t patchH, int64_t patchW, int64_t padH, int64_t padW,
                         int64_t dilationH, int64_t dilationW, int64_t dilation_patchH,
                         int64_t dilation_patchW, int64_t dH, int64_t dW);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiCorrelationBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiTensorHandle_t input1, diopiTensorHandle_t input2,
                          diopiTensorHandle_t grad_input1, diopiTensorHandle_t grad_input2, int64_t kH,
                          int64_t kW, int64_t patchH, int64_t patchW, int64_t padH, int64_t padW,
                          int64_t dilationH, int64_t dilationW, int64_t dilation_patchH,
                          int64_t dilation_patchW, int64_t dH, int64_t dW);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiDeformableIm2col(diopiContextHandle_t ctx, diopiTensorHandle_t data_im, diopiTensorHandle_t data_offset, diopiTensorHandle_t data_col,
                            int64_t channels, int64_t height,
                            int64_t width, int64_t ksize_h,
                            int64_t ksize_w, int64_t pad_h, int64_t pad_w,
                            int64_t stride_h, int64_t stride_w,
                            int64_t dilation_h, int64_t dilation_w,
                            int64_t parallel_imgs, int64_t deformable_group);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiDeformableCol2im(diopiContextHandle_t ctx, diopiTensorHandle_t data_col, diopiTensorHandle_t data_offset, diopiTensorHandle_t grad_im,
                            int64_t channels, int64_t height,
                            int64_t width, int64_t ksize_h,
                            int64_t ksize_w, int64_t pad_h, int64_t pad_w,
                            int64_t stride_h, int64_t stride_w,
                            int64_t dilation_h, int64_t dilation_w,
                            int64_t parallel_imgs, int64_t deformable_group);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiDeformableCol2imCoord(diopiContextHandle_t ctx, diopiTensorHandle_t data_col, diopiTensorHandle_t data_im, diopiTensorHandle_t data_offset, int64_t channels, diopiTensorHandle_t grad_offset,
                                            int64_t height, int64_t width, int64_t ksize_h, int64_t ksize_w,
                                            int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w,
                                            int64_t dilation_h, int64_t dilation_w, int64_t parallel_imgs,
                                            int64_t deformable_group);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiDeformRoiPool(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t rois, diopiTensorHandle_t offset,
                                  diopiTensorHandle_t output, int64_t pooled_height,
                                  int64_t pooled_width, float spatial_scale,
                                  int64_t sampling_ratio, float gamma);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiDeformRoiPoolBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiTensorHandle_t input,
                                   diopiTensorHandle_t rois, diopiTensorHandle_t offset,
                                   diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_offset,
                                   int64_t pooled_height, int64_t pooled_width,
                                   float spatial_scale, int64_t sampling_ratio,
                                   float gamma);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiSigmoidFocalLossMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t target,
                                            diopiTensorHandle_t weight, diopiTensorHandle_t output,
                                            float gamma,
                                            float alpha);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiSigmoidFocalLossBackwardMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t target,
                                                    diopiTensorHandle_t weight,
                                                    diopiTensorHandle_t grad_input,
                                                    float gamma,
                                                    float alpha);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiSoftmaxFocalLoss(diopiContextHandle_t ctx, diopiTensorHandle_t softmax, diopiTensorHandle_t target,
                                            diopiTensorHandle_t weight, diopiTensorHandle_t output,
                                            float gamma,
                                            float alpha);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiSoftmaxFocalLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t softmax, diopiTensorHandle_t target,
                                                    diopiTensorHandle_t weight, diopiTensorHandle_t buff,
                                                    diopiTensorHandle_t grad_input,
                                                    float gamma,
                                                    float alpha);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiFurthestPointSampling(diopiContextHandle_t ctx, diopiTensorHandle_t points_tensor,
                                                diopiTensorHandle_t temp_tensor, diopiTensorHandle_t idx_tensor,
                                                int64_t b, int64_t n, int64_t m);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiFurthestPointSamplingWithDist(diopiContextHandle_t ctx, diopiTensorHandle_t points_tensor,
                                                        diopiTensorHandle_t temp_tensor,
                                                        diopiTensorHandle_t idx_tensor, int64_t b,
                                                        int64_t n, int64_t m);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiFusedBiasLeakyrelu(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input,
                                               diopiConstTensorHandle_t bias, diopiConstTensorHandle_t refer, int64_t act, int64_t grad, float alpha, float scale);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiGatherPoints(diopiContextHandle_t ctx, diopiConstTensorHandle_t points, diopiConstTensorHandle_t idx, diopiTensorHandle_t out,
                                                  int64_t b, int64_t c, int64_t n, int64_t npoints);
/**
 * \brief
 */
DIOPI_API diopiError_t diopiGatherPointsBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t idx, diopiTensorHandle_t grad_points,
                                                   int64_t b, int64_t c, int64_t n, int64_t npoints);
/**
 * \brief
 */
DIOPI_API diopiError_t diopiGroupPoints(diopiContextHandle_t ctx, diopiConstTensorHandle_t points, diopiConstTensorHandle_t idx, diopiTensorHandle_t out,
                                                 int64_t b, int64_t c, int64_t n, int64_t npoints, int64_t nsample);
/**
 * \brief
 */
DIOPI_API diopiError_t diopiGroupPointsBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t idx, diopiTensorHandle_t grad_points,
                                                  int64_t b, int64_t c, int64_t n, int64_t npoints, int64_t nsample);
/**
 * \brief
 */
DIOPI_API diopiError_t diopiStackGroupPoints(diopiContextHandle_t ctx,
                                     diopiConstTensorHandle_t features_tensor,
                                     diopiConstTensorHandle_t features_batch_cnt_tensor,
                                     diopiConstTensorHandle_t idx_tensor,
                                     diopiConstTensorHandle_t idx_batch_cnt_tensor,
                                     diopiTensorHandle_t out_tensor,
                                     int64_t b, int64_t c, int64_t m, int64_t nsample);
/**
 * \brief
 */
DIOPI_API diopiError_t diopiStackGroupPointsBackward(diopiContextHandle_t ctx,
                                      diopiConstTensorHandle_t grad_out_tensor,
                                      diopiConstTensorHandle_t idx_tensor,
                                      diopiConstTensorHandle_t idx_batch_cnt_tensor,
                                      diopiConstTensorHandle_t features_batch_cnt_tensor,
                                      diopiTensorHandle_t grad_features_tensor,
                                      int64_t b, int64_t c, int64_t m, int64_t n, int64_t nsample);
/**
 * \brief
 */
DIOPI_API diopiError_t diopiIou3dBoxesOverlapBev(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes_a, diopiConstTensorHandle_t boxes_b,
                                                            diopiTensorHandle_t ans_overlap,int64_t num_a, int64_t num_b);
/**
 * \brief
 */
DIOPI_API diopiError_t diopiIou3dNms3d(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes, diopiTensorHandle_t keep, diopiTensorHandle_t keep_num, float nms_overlap_thresh);
/**
 * \brief
 */
DIOPI_API diopiError_t diopiIou3dNms3dNormal(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes, diopiTensorHandle_t keep, diopiTensorHandle_t keep_num, float nms_overlap_thresh);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiKnn(diopiContextHandle_t ctx, diopiTensorHandle_t xyz_tensor, diopiTensorHandle_t new_xyz_tensor, diopiTensorHandle_t idx_tensor,
                 diopiTensorHandle_t dist2_tensor, int64_t b, int64_t n, int64_t m, int64_t nsample);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiMaskedIm2col(diopiContextHandle_t ctx, diopiConstTensorHandle_t im, diopiConstTensorHandle_t mask_h_idx,
                                diopiConstTensorHandle_t mask_w_idx, diopiTensorHandle_t col,
                                int64_t kernel_h, int64_t kernel_w,
                                int64_t pad_h, int64_t pad_w);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiMaskedCol2im(diopiContextHandle_t ctx, diopiConstTensorHandle_t col, diopiConstTensorHandle_t mask_h_idx,
                                diopiConstTensorHandle_t mask_w_idx, diopiTensorHandle_t im, int64_t height,
                                int64_t width, int64_t channels);
/**
 * \brief
 */
DIOPI_API diopiError_t diopiModulatedDeformableIm2col(
    diopiContextHandle_t ctx, diopiConstTensorHandle_t data_im, diopiConstTensorHandle_t data_offset, diopiConstTensorHandle_t data_mask,
    int64_t batch_size, int64_t channels, int64_t height_im,
    int64_t width_im, int64_t height_col, int64_t width_col,
    int64_t kernel_h, int64_t kernel_w, int64_t pad_h, int64_t pad_w,
    int64_t stride_h, int64_t stride_w, int64_t dilation_h,
    int64_t dilation_w, int64_t deformable_group, diopiTensorHandle_t data_col);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiModulatedDeformableCol2im(
    diopiContextHandle_t ctx, diopiConstTensorHandle_t data_col, diopiConstTensorHandle_t data_offset, diopiConstTensorHandle_t data_mask,
    int64_t batch_size, int64_t channels, int64_t height_im,
    int64_t width_im, int64_t height_col, int64_t width_col,
    int64_t kernel_h, int64_t kernel_w, int64_t pad_h, int64_t pad_w,
    int64_t stride_h, int64_t stride_w, int64_t dilation_h,
    int64_t dilation_w, int64_t deformable_group, diopiTensorHandle_t grad_im);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiModulatedDeformableCol2imCoord(
    diopiContextHandle_t ctx, diopiConstTensorHandle_t data_col, diopiConstTensorHandle_t data_im, diopiConstTensorHandle_t data_offset,
    diopiConstTensorHandle_t data_mask, int64_t batch_size, int64_t channels,
    int64_t height_im, int64_t width_im, int64_t height_col,
    int64_t width_col, int64_t kernel_h, int64_t kernel_w,
    int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w,
    int64_t dilation_h, int64_t dilation_w, int64_t deformable_group,
    diopiTensorHandle_t grad_offset, diopiTensorHandle_t grad_mask);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiModulatedDeformableIm2col(
    diopiContextHandle_t ctx, diopiConstTensorHandle_t data_im, diopiConstTensorHandle_t data_offset, diopiConstTensorHandle_t data_mask,
    int64_t batch_size, int64_t channels, int64_t height_im,
    int64_t width_im, int64_t height_col, int64_t width_col,
    int64_t kernel_h, int64_t kernel_w, int64_t pad_h, int64_t pad_w,
    int64_t stride_h, int64_t stride_w, int64_t dilation_h,
    int64_t dilation_w, int64_t deformable_group, diopiTensorHandle_t data_col);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiModulatedDeformableCol2im(
    diopiContextHandle_t ctx, diopiConstTensorHandle_t data_col, diopiConstTensorHandle_t data_offset, diopiConstTensorHandle_t data_mask,
    int64_t batch_size, int64_t channels, int64_t height_im,
    int64_t width_im, int64_t height_col, int64_t width_col,
    int64_t kernel_h, int64_t kernel_w, int64_t pad_h, int64_t pad_w,
    int64_t stride_h, int64_t stride_w, int64_t dilation_h,
    int64_t dilation_w, int64_t deformable_group, diopiTensorHandle_t grad_im);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiModulatedDeformableCol2imCoord(
    diopiContextHandle_t ctx, diopiConstTensorHandle_t data_col, diopiConstTensorHandle_t data_im, diopiConstTensorHandle_t data_offset,
    diopiConstTensorHandle_t data_mask, int64_t batch_size, int64_t channels,
    int64_t height_im, int64_t width_im, int64_t height_col,
    int64_t width_col, int64_t kernel_h, int64_t kernel_w,
    int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w,
    int64_t dilation_h, int64_t dilation_w, int64_t deformable_group,
    diopiTensorHandle_t grad_offset, diopiTensorHandle_t grad_mask);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiMsDeformAttn(diopiContextHandle_t ctx, diopiTensorHandle_t* out,
                                   diopiConstTensorHandle_t value,
                                   diopiConstTensorHandle_t spatial_shapes,
                                   diopiConstTensorHandle_t level_start_index,
                                   diopiConstTensorHandle_t sampling_loc,
                                   diopiConstTensorHandle_t attn_weight,
                                   int64_t im2col_step);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiMsDeformAttnBackward(
    diopiContextHandle_t ctx, diopiConstTensorHandle_t value, diopiConstTensorHandle_t spatial_shapes,
    diopiConstTensorHandle_t level_start_index, diopiConstTensorHandle_t sampling_loc,
    diopiConstTensorHandle_t attn_weight, diopiConstTensorHandle_t grad_output, diopiTensorHandle_t grad_value,
    diopiTensorHandle_t grad_sampling_loc, diopiTensorHandle_t grad_attn_weight, int64_t im2col_step);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiNmsMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t dets,
                                    diopiConstTensorHandle_t scores, double iou_threshold, int64_t offset);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiPointsInBoxesPart(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes, diopiConstTensorHandle_t pts, diopiTensorHandle_t box_idx_of_points,
                                                    int64_t batch_size, int64_t boxes_num, int64_t pts_num);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiPointsInBoxesAll(diopiContextHandle_t ctx, int64_t batch_size, int64_t boxes_num,
                                      int64_t pts_num, diopiConstTensorHandle_t boxes,
                                      diopiConstTensorHandle_t pts,
                                      diopiTensorHandle_t box_idx_of_points);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiPsamask(diopiContextHandle_t ctx, int64_t psa_type, diopiConstTensorHandle_t input, diopiTensorHandle_t output,
                          int64_t num_, int64_t h_feature,
                          int64_t w_feature, int64_t h_mask,
                          int64_t w_mask, int64_t half_h_mask,
                          int64_t half_w_mask);
/**
 * \brief
 */
DIOPI_API diopiError_t diopiPsamaskBackward(diopiContextHandle_t ctx, int64_t psa_type, diopiConstTensorHandle_t grad_output,
                           diopiTensorHandle_t grad_input, int64_t num_,
                           int64_t h_feature, int64_t w_feature,
                           int64_t h_mask, int64_t w_mask,
                           int64_t half_h_mask, int64_t half_w_mask);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiRoiAlignMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t rois, diopiTensorHandle_t output,
                            diopiTensorHandle_t argmax_y, diopiTensorHandle_t argmax_x,
                            int64_t aligned_height, int64_t aligned_width,
                            float spatial_scale, int64_t sampling_ratio,
                            int64_t pool_mode, bool aligned);
/**
 * \brief
 */
DIOPI_API diopiError_t diopiRoiAlignBackwardMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiTensorHandle_t rois, diopiTensorHandle_t argmax_y,
                             diopiTensorHandle_t argmax_x, diopiTensorHandle_t grad_input,
                             int64_t aligned_height, int64_t aligned_width,
                             float spatial_scale, int64_t sampling_ratio,
                             int64_t pool_mode, bool aligned);
/**
 * \brief
 */
DIOPI_API diopiError_t diopiRoiAlignRotated(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t rois, diopiTensorHandle_t output,
                                    int64_t aligned_height, int64_t aligned_width,
                                    float spatial_scale, int64_t sampling_ratio,
                                    bool aligned, bool clockwise);
/**
 * \brief
 */
DIOPI_API diopiError_t diopiRoiAlignRotatedBackward(diopiContextHandle_t ctx, diopiTensorHandle_t top_grad, diopiTensorHandle_t rois,
                                     diopiTensorHandle_t bottom_grad, int64_t aligned_height,
                                     int64_t aligned_width, float spatial_scale,
                                     int64_t sampling_ratio, bool aligned,
                                     bool clockwise);
/**
 * \brief
 */
DIOPI_API diopiError_t diopiRiroiAlignRotated(diopiContextHandle_t ctx, diopiTensorHandle_t features, diopiTensorHandle_t rois,
                                      diopiTensorHandle_t output, int64_t pooled_height,
                                      int64_t pooled_width, float spatial_scale,
                                      int64_t num_samples, int64_t num_orientations,
                                      bool clockwise);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiRiroiAlignRotatedBackward(diopiContextHandle_t ctx, diopiTensorHandle_t top_grad, diopiTensorHandle_t rois,
                                       diopiTensorHandle_t bottom_grad, int64_t pooled_height,
                                       int64_t pooled_width, float spatial_scale,
                                       int64_t num_samples, int64_t num_orientations,
                                       bool clockwise);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiRoiawarePool3d(diopiContextHandle_t ctx, int64_t boxes_num, int64_t pts_num, int64_t channels,
                                  int64_t max_pts_each_voxel, int64_t out_x, int64_t out_y,
                                  int64_t out_z, diopiConstTensorHandle_t rois,
                                  diopiConstTensorHandle_t pts, diopiConstTensorHandle_t pts_feature,
                                  diopiTensorHandle_t argmax, diopiTensorHandle_t pts_idx_of_voxels,
                                  diopiTensorHandle_t pooled_features, int64_t pool_method);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiRoiawarePool3dBackward(diopiContextHandle_t ctx, int64_t boxes_num, int64_t out_x, int64_t out_y,
                                   int64_t out_z, int64_t channels,
                                   int64_t max_pts_each_voxel,
                                   diopiConstTensorHandle_t pts_idx_of_voxels,
                                   diopiConstTensorHandle_t argmax, diopiConstTensorHandle_t grad_out,
                                   diopiTensorHandle_t grad_in, int64_t pool_method);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiRoipointPool3d(diopiContextHandle_t ctx, int64_t batch_size, int64_t pts_num, int64_t boxes_num,
                                  int64_t feature_in_len, int64_t sampled_pts_num,
                                  diopiConstTensorHandle_t xyz, diopiConstTensorHandle_t boxes3d,
                                  diopiConstTensorHandle_t pts_feature,
                                  diopiTensorHandle_t pooled_features,
                                  diopiTensorHandle_t pooled_empty_flag);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiRoiPool(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t rois, diopiTensorHandle_t output,
                           diopiTensorHandle_t argmax, int64_t pooled_height, int64_t pooled_width,
                           float spatial_scale);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiRoiPoolBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiTensorHandle_t rois, diopiTensorHandle_t argmax,
                            diopiTensorHandle_t grad_input, int64_t pooled_height,
                            int64_t pooled_width, float spatial_scale);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiMinAreaPolygons(diopiContextHandle_t ctx, diopiConstTensorHandle_t pointsets, diopiTensorHandle_t polygons);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiActiveRotatedFilter(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices, diopiTensorHandle_t output);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiActiveRotatedFilterBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t indices, diopiTensorHandle_t grad_in);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiConvexIou(diopiContextHandle_t ctx, diopiConstTensorHandle_t pointsets, diopiConstTensorHandle_t polygons, diopiTensorHandle_t ious);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiConvexGiou(diopiContextHandle_t ctx, diopiConstTensorHandle_t pointsets, diopiConstTensorHandle_t polygons, diopiTensorHandle_t output);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiDiffIouRotatedSortVertices(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiTensorHandle_t vertices, diopiTensorHandle_t mask,
                                                   diopiTensorHandle_t num_valid);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiChamferDistance(diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz1, diopiConstTensorHandle_t xyz2,
                                            diopiTensorHandle_t dist1, diopiTensorHandle_t dist2, diopiTensorHandle_t idx1, diopiTensorHandle_t idx2);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiChamferDistanceBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz1, diopiConstTensorHandle_t xyz2,
                                            diopiConstTensorHandle_t idx1, diopiConstTensorHandle_t idx2, diopiConstTensorHandle_t grad_dist1, diopiConstTensorHandle_t grad_dist2,
                                            diopiTensorHandle_t grad_xyz1, diopiTensorHandle_t grad_xyz2);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiPrroiPool(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t rois, diopiTensorHandle_t output,
                                      int64_t pooled_height, int64_t pooled_width, float spatial_scale);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiPrroiPoolbackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiTensorHandle_t rois, diopiTensorHandle_t grad_input,
                                              int64_t pooled_height, int64_t pooled_width, float spatial_scale);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiPrroiPoolCoorBackward(diopiContextHandle_t ctx, diopiTensorHandle_t output, diopiTensorHandle_t grad_output, diopiTensorHandle_t input, diopiTensorHandle_t rois, diopiTensorHandle_t grad_rois,
                                                  int64_t pooled_height, int64_t pooled_width, float spatial_scale);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_H_
