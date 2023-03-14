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
DIOPI_API diopiError_t diopiAddcmulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value);

DIOPI_API diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                   diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value);
DIOPI_API diopiError_t diopiAddcdivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value);

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
 * \brief Computes the element-wise logical AND/OR/NOT of the given input tensors.
 */
DIOPI_API diopiError_t diopiLogicalAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiLogicalAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t diopiLogicalOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiLogicalOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t diopiLogicalNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
DIOPI_API diopiError_t diopiLogicalNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * \brief Computes the bitwise AND/OR/NOT of the given input tensors.
 */
DIOPI_API diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiBitwiseAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiBitwiseAndInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

DIOPI_API diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiBitwiseOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiBitwiseOrInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

DIOPI_API diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
DIOPI_API diopiError_t diopiBitwiseNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

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
DIOPI_API diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim);

/**
 * \brief Returns True if all elements in each row of the tensor in the given dimension dim are True, False otherwise.
 */
DIOPI_API diopiError_t diopiAll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim);

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
DIOPI_API diopiError_t diopiClipGradNorm(diopiContextHandle_t ctx, double* out, diopiTensorHandle_t *grads,
                                         int64_t num_grads, double max_norm, double norm_type, bool error_if_nonfinite);

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
 * \brief Implements Rmsprop optimizer.
 */
DIOPI_API diopiError_t diopiRmsprop(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg, diopiTensorHandle_t grad_avg,
                                    diopiTensorHandle_t momentum_buf, float lr, float alpha, float eps, float weight_decay, float momentum, bool centered);

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
DIOPI_API diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, const int64_t* dim,
                                   bool sorted, bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts);

/**
 * \brief Returns the product of all elements in the input tensor.
 */
DIOPI_API diopiError_t diopiProd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, diopiDtype_t type);

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
DIOPI_API diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t other);

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

DIOPI_API diopiError_t diopiNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std);
DIOPI_API diopiError_t diopiNormalTensorScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, double std);
DIOPI_API diopiError_t diopiNormalScalarTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, diopiConstTensorHandle_t std);
DIOPI_API diopiError_t diopiNormalTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t std);

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

/**
 * \brief Repeats tensor input along the specified dimensions.
 */
DIOPI_API diopiError_t diopiRepeat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeats_size);







/* DIOPI functions from MMCV extension ops <https://github.com/open-mmlab/mmcv.git> */

/**
 * @brief Perform weighted sum to generate output features according to scores.
 * @param[in] ctx diopi context.
 * @param scores (B, npoint, K, M), predicted scores to
          aggregate weight matrices in the weight bank.
          ``npoint`` is the number of sampled centers.
          ``K`` is the number of queried neighbors.
          ``M`` is the number of weight matrices in the weight bank.
 * @param points (B, N, M, out_dim)
         Pre-computed point features to be aggregated.
 * @param centers (B, N, M, out_dim)
         Pre-computed center features to be aggregated.
 * @param knn_idx (B, npoint, K), index of sampled kNN.
         We assume the first idx in each row is the idx of the center.
 * @param B,N,npoint,M,K,out_dim scores: (B, npoint, K, M); points: (B, N, M, out_dim)
 * @param aggregate (str, optional): Aggregation method. Can be 'sum', 'avg' or 'max'. Defaults: 'sum'.
 * @param[out] output (B, out_dim, npoint, K), the aggregated features.
 */
DIOPI_API diopiError_t diopiAssignScoreWithk(diopiContextHandle_t ctx, diopiConstTensorHandle_t points, diopiConstTensorHandle_t centers,
                                             diopiConstTensorHandle_t scores, diopiConstTensorHandle_t knn_idx,
                                             diopiTensorHandle_t output, int64_t B, int64_t N, int64_t npoint, int64_t M,
                                             int64_t K, int64_t out_dim, int64_t aggregate);
/**
 * @brief Backward function for perform weighted sum to generate output features according to scores.
 * @param[in] ctx diopi context.
 * @param grad_out the gradient of ``output``
 * @param[out] grad_points the gradient of ``points``.
 * @param grad_scores the gradient of ``scores``.
 * @param grad_centers the gradient of ``centers``
 * @sa definition of other parameters, refer to diopiAssignScoreWithk()
 */
DIOPI_API diopiError_t diopiAssignScoreWithkBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t points,
                                                     diopiConstTensorHandle_t centers, diopiConstTensorHandle_t scores,
                                                     diopiConstTensorHandle_t knn_idx, diopiTensorHandle_t grad_points,
                                                     diopiTensorHandle_t grad_centers, diopiTensorHandle_t grad_scores,
                                                     int64_t B, int64_t N, int64_t npoint, int64_t M, int64_t K, int64_t out_dim,
                                                     int64_t aggregate);

/**
 * @brief Find nearby points in spherical space.
 * @param[in] ctx diopi context.
 * @param min_radius float: minimum radius of the balls.
 * @param max_radius float: maximum radius of the balls.
 * @param sample_num int: maximum number of features in the balls.
 * @param xyz (B, N, 3) xyz coordinates of the features,
                 or staked input (N1 + N2 ..., 3).
 * @param center_xyz (B, npoint, 3) centers of the ball
     query, or staked input (M1 + M2 ..., 3).
 * @param B,N,npoint xyz (B, N, 3), center_xyz (B, npoint, 3)
 * @param[out] idx (B, npoint, nsample) tensor with the indices of the
     features that form the query balls.
 */
DIOPI_API diopiError_t diopiBallQuery(diopiContextHandle_t ctx, diopiConstTensorHandle_t center_xyz, diopiConstTensorHandle_t xyz, diopiTensorHandle_t idx, int64_t B, int64_t N, int64_t npoint,
                                      float min_radius, float max_radius, int64_t sample_num);
/**
 * @brief Find nearby points in spherical space.(Stacked method)
 * @param[in] ctx diopi context.
 * @param xyz_batch_cnt (batch_size): Stacked input xyz coordinates nums in
     each batch, just like (N1, N2, ...). Defaults to None.
     New in version 1.7.0.
 * @param center_xyz_batch_cnt (batch_size): Stacked centers coordinates
     nums in each batch, just line (M1, M2, ...). Defaults to None.
     New in version 1.7.0.
 * @sa definition of other parameters, refer to diopiBallQuery()
 */
DIOPI_API diopiError_t diopiStackBallQuery(diopiContextHandle_t ctx, diopiConstTensorHandle_t center_xyz, diopiConstTensorHandle_t center_xyz_batch_cnt, diopiConstTensorHandle_t xyz,
                                           diopiConstTensorHandle_t xyz_batch_cnt, diopiTensorHandle_t idx, float max_radius, int64_t sample_num);

/**
 * @brief Calculate overlap between two set of bboxes.
 * @param[in] ctx diopi context.
 * @param bboxes1 shape (m, 4) in <x1, y1, x2, y2> format or
     empty.
 * @param bboxes2 shape (n, 4) in <x1, y1, x2, y2> format or
     empty. If aligned is ``True``, then m and n must be equal.
 * @param mode (str): "iou" (intersection over union) or iof (intersection over
     foreground).
 * @param aligned bool, determine the way how ``ious`` is calculated.
 * @param offset  offset is 0 or 1.
 * @param[out] ious  the ious betweens boxes. If ``aligned`` is
     ``False``, the shape of ious is (m, n) else (m, 1).
 */
DIOPI_API diopiError_t diopiBboxOverlaps(diopiContextHandle_t ctx, diopiConstTensorHandle_t bboxes1, diopiConstTensorHandle_t bboxes2, diopiTensorHandle_t ious,
                                         int64_t mode, bool aligned, int64_t offset);

/**
 * @brief Border align pooling layer.
 * @param[in] ctx diopi context.
 * @param input  Features with shape [N,4C,H,W]. Channels ranged in [0,C),
     [C,2C), [2C,3C), [3C,4C) represent the top, left, bottom,
     right features respectively.
 * @param boxes  Boxes with shape [N,H*W,4]. Coordinate format (x1,y1,x2,y2).
 * @param pool_size int: number of positions sampled over the boxes' borders
         (e.g. top, bottom, left, right).
 * @param[out] output  Pooled features with shape [N,C,H*W,4]. The order is
         (top,left,bottom,right) for the last dimension.
 * @param argmax_idx  `argmax_idx` only used for backward.
 */
DIOPI_API diopiError_t diopiBorderAlign(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t boxes,
                                        diopiTensorHandle_t output, diopiTensorHandle_t argmax_idx,
                                        int64_t pool_size);
/**
 * @brief Backward function for border align pooling layer.
 * @param[in] ctx diopi context.
 * @param grad_output the gradient of ``output``
 * @param[out] grad_input  the gradient of ``input``
 * @sa definition of other parameters, refer to diopiBorderAlign()
 */
DIOPI_API diopiError_t diopiBorderAlignBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t boxes,
                                                diopiConstTensorHandle_t argmax_idx, diopiTensorHandle_t grad_input,
                                                int64_t pool_size);

/**
 * @brief Return intersection-over-union (Jaccard index) of boxes(BoxIouRotated).
 * @param[in] ctx diopi context.
 * @param bboxes1   quadrilateral bboxes 1. It has shape (N, 8),
     indicating (x1, y1, ..., x4, y4) for each row.
 * @param bboxes2   quadrilateral bboxes 2. It has shape (M, 8),
     indicating (x1, y1, ..., x4, y4) for each row.
 * @param mode (str)  "iou" (intersection over union) or iof (intersection over
     foreground).
 * @param aligned  If ``aligned`` is ``False``, then calculate the ious between each bbox
     of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
     bboxes1 and bboxes2.
 * @param[out] ious  the ious betweens boxes. If ``aligned`` is ``False``,
     the shape of ious is (N, M) else (N,).
 */
DIOPI_API diopiError_t diopiBoxIouRotated(diopiContextHandle_t ctx, diopiConstTensorHandle_t bboxes1, diopiConstTensorHandle_t bboxes2, diopiTensorHandle_t ious,
                                          int64_t mode, bool aligned);
/**
 * @brief Return intersection-over-union (Jaccard index) of boxes(BoxIouQuadri).
 * @param[in] ctx diopi context.
 * @sa definition of other parameters, refer to diopiBoxIouRotated()
 */
DIOPI_API diopiError_t diopiBoxIouQuadri(diopiContextHandle_t ctx, diopiConstTensorHandle_t bboxes1, diopiConstTensorHandle_t bboxes2, diopiTensorHandle_t ious,
                                         int64_t mode, bool aligned);

/**
 * @brief Content-Aware ReAssembly of FEatures. Please refer to `CARAFE: Content-Aware ReAssembly of FEatures
    <https://arxiv.org/abs/1905.02188>`_ for more details. 
 */
DIOPI_API diopiError_t diopiCarafe(diopiContextHandle_t ctx, diopiConstTensorHandle_t features, diopiConstTensorHandle_t masks, diopiTensorHandle_t rfeatures,
                                   diopiTensorHandle_t routput, diopiTensorHandle_t rmasks, diopiTensorHandle_t output, int64_t kernel_size, int64_t group_size, int64_t scale_factor);
DIOPI_API diopiError_t diopiCarafeBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t top_grad, diopiConstTensorHandle_t rfeatures, diopiConstTensorHandle_t masks,
                                           diopiTensorHandle_t rtop_grad, diopiTensorHandle_t rbottom_grad_hs, diopiTensorHandle_t rbottom_grad, diopiTensorHandle_t rmask_grad,
                                           diopiTensorHandle_t bottom_grad, diopiTensorHandle_t mask_grad, int64_t kernel_size, int64_t group_size, int64_t scale_factor);
DIOPI_API diopiError_t diopiCarafeNaive(diopiContextHandle_t ctx, diopiConstTensorHandle_t features,
                                        diopiConstTensorHandle_t masks, diopiTensorHandle_t output,
                                        int64_t kernel_size,
                                        int64_t group_size,
                                        int64_t scale_factor);
DIOPI_API diopiError_t diopiCarafeNaiveBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t top_grad, diopiConstTensorHandle_t features, diopiConstTensorHandle_t masks,
                                                diopiTensorHandle_t bottom_grad, diopiTensorHandle_t mask_grad, int64_t kernel_size, int64_t group_size, int64_t scale_factor);


/**
 * @brief This correlation operator works for optical flow correlation computation.
 * @param[in] ctx diopi context.
 * @param input1, input2  input tensor
 * @param kH, kW kernel_size int: The size of sliding window i.e. local neighborhood
     representing the center points and involved in correlation
     computation. Defaults to 1. kH, kW = _pair(kernel_size).
 * @param patchH, patchW max_displacement int: The radius for computing correlation volume,
     but the actual working space can be dilated by dilation_patch.
     Defaults to 1. patch_size = max_displacement * 2 + 1.
     patchH, patchW = _pair(patch_size)
 * @param dH, dW stride int: The stride of the sliding blocks in the input spatial
     dimensions. Defaults to 1. dH, dW = _pair(stride)
 * @param padH, padW padding int: Zero padding added to all four sides of the input1.
     Defaults to 0. padH, padW = _pair(padding).
 * @param dilationH, dilationW dilation int: The spacing of local neighborhood that will involved
     in correlation. Defaults to 1. dilationH, dilationW = _pair(dilation).
 * @param dilation_patchH, dilation_patchW dilation_patch int: The spacing between position need to compute
     correlation.  Defaults to 1. dilation_patchH, dilation_patchW =  _pair(
     dilation_patch)
 * @param[out] output  correlation operator out tensor for input tensor
 */
DIOPI_API diopiError_t diopiCorrelation(diopiContextHandle_t ctx, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, diopiTensorHandle_t output, int64_t kH,
                                        int64_t kW, int64_t patchH, int64_t patchW, int64_t padH, int64_t padW,
                                        int64_t dilationH, int64_t dilationW, int64_t dilation_patchH,
                                        int64_t dilation_patchW, int64_t dH, int64_t dW);
/**
 * @brief Backward function for the correlation operator works for optical flow correlation computation.
 * @param[in] ctx diopi context.
 * @param grad_output the gradient of ``output`` 
 * @param[out] grad_input1, grad_input2  the gradient of input tensors
 * @sa definition of other parameters, refer to diopiCorrelation()
 */
DIOPI_API diopiError_t diopiCorrelationBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2,
                                                diopiTensorHandle_t grad_input1, diopiTensorHandle_t grad_input2, int64_t kH,
                                                int64_t kW, int64_t patchH, int64_t patchW, int64_t padH, int64_t padW,
                                                int64_t dilationH, int64_t dilationW, int64_t dilation_patchH,
                                                int64_t dilation_patchW, int64_t dH, int64_t dW);

/**
 * @brief Deformable 2D convolution.
 * @param[in] ctx diopi context.
 * @param input  Input feature, shape (B, C_in, H_in, W_in)
 * @param weight  weight tensor,
 * @param offset  Offset for deformable convolution, shape
         (B, deform_groups*kernel_size[0]*kernel_size[1]*2,
         H_out, W_out), H_out, W_out are equal to the output's.
         kernel_size(int, tuple): Size of the convolving kernel.
         An offset is like `[y0, x0, y1, x1, y2, x2, ..., y8, x8]`.
         The spatial arrangement is like:
         .. code:: text
             (x0, y0) (x1, y1) (x2, y2)
             (x3, y3) (x4, y4) (x5, y5)
             (x6, y6) (x7, y7) (x8, y8)
 * @param kH, kW kH=weight.size(2),kW=weight.size(3)
 * @param dH, dW stride (int, tuple): Stride of the convolution. Default: 1.
     dW=stride[1], dH=stride[0]
 * @param padH, padW padding (int or tuple): Zero-padding added to both sides of the input.
     Default: 0. padW=padding[1], padH=padding[0],
 * @param dilationH,dilationW dilation (int or tuple): Spacing between kernel elements. Default: 1.
     dilationW= dilation[1], dilationH= dilation[0]
 * @param groups int: Number of blocked connections from input.
     channels to output channels. Default: 1.
 * @param deform_groups int: Number of deformable group partitions.
 * @param im2col_step int: Number of samples processed by im2col_cuda_kernel
     per call. It will work when ``batch_size`` > ``im2col_step``, but
     ``batch_size`` must be divisible by ``im2col_step``. Default: 32.
     `New in version 1.3.17.`
 * @param[out] output, columns, ones  Output of the layer.
 */
DIOPI_API diopiError_t diopiDeformConv(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t offset,
                                       diopiTensorHandle_t output, diopiTensorHandle_t columns, diopiTensorHandle_t ones, int64_t kW,
                                       int64_t kH, int64_t dW, int64_t dH, int64_t padW, int64_t padH,
                                       int64_t dilationW, int64_t dilationH, int64_t groups,
                                       int64_t deform_groups, int64_t im2col_step);
/**
 * @brief Backward function for Deformable 2D convolution(for input and offset).
 * @param[in] ctx diopi context.
 * @param gradOutput  the gradient of ``output``.
 * @param[out] gradInput, gradOffset  the gradient of input tensors ``input`` and ``offset``
 * @sa definition of other parameters, refer to diopiDeformConv()
 */
DIOPI_API diopiError_t diopiDeformConvBackwardInput(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t offset, diopiConstTensorHandle_t gradOutput,
                                                    diopiTensorHandle_t gradInput, diopiTensorHandle_t gradOffset,
                                                    diopiConstTensorHandle_t weight, diopiConstTensorHandle_t columns, int64_t kW, int64_t kH,
                                                    int64_t dW, int64_t dH, int64_t padW, int64_t padH,
                                                    int64_t dilationW, int64_t dilationH, int64_t groups,
                                                    int64_t deform_groups, int64_t im2col_step);
/**
 * @brief Backward function for Deformable 2D convolution(for weight).
 * @param[in] ctx diopi context.
 * @param gradOutput  the gradient of ``output``.
 * @param[out] gradWeight  the gradient of input tensors ``weight``
 * @sa definition of other parameters, refer to diopiDeformConv()
 */
DIOPI_API diopiError_t diopiDeformConvBackwardParameters(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t offset,
                                                         diopiTensorHandle_t gradOutput, diopiTensorHandle_t gradWeight,
                                                         diopiConstTensorHandle_t columns, diopiConstTensorHandle_t ones, int64_t kW,
                                                         int64_t kH, int64_t dW, int64_t dH, int64_t padW, int64_t padH,
                                                         int64_t dilationW, int64_t dilationH, int64_t groups,
                                                         int64_t deform_groups, float scale,
                                                         int64_t im2col_step);

/**
 * @brief Deformable RoiPool.
 */
DIOPI_API diopiError_t diopiDeformRoiPool(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t rois, diopiTensorHandle_t offset,
                                          diopiTensorHandle_t output, int64_t pooled_height,
                                          int64_t pooled_width, float spatial_scale,
                                          int64_t sampling_ratio, float gamma);
DIOPI_API diopiError_t diopiDeformRoiPoolBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiTensorHandle_t input,
                                                  diopiTensorHandle_t rois, diopiTensorHandle_t offset,
                                                  diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_offset,
                                                  int64_t pooled_height, int64_t pooled_width,
                                                  float spatial_scale, int64_t sampling_ratio,
                                                  float gamma);

/**
 * @brief SigmoidFocalLoss
 */
DIOPI_API diopiError_t diopiSigmoidFocalLossMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t target,
                                                 diopiTensorHandle_t weight, diopiTensorHandle_t output,
                                                 float gamma,
                                                 float alpha);
DIOPI_API diopiError_t diopiSigmoidFocalLossBackwardMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t target,
                                                         diopiTensorHandle_t weight,
                                                         diopiTensorHandle_t grad_input,
                                                         float gamma,
                                                         float alpha);

/**
 * @brief SoftmaxFocalLoss
 */
DIOPI_API diopiError_t diopiSoftmaxFocalLoss(diopiContextHandle_t ctx, diopiTensorHandle_t softmax, diopiTensorHandle_t target,
                                             diopiTensorHandle_t weight, diopiTensorHandle_t output,
                                             float gamma,
                                             float alpha);
DIOPI_API diopiError_t diopiSoftmaxFocalLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t softmax, diopiTensorHandle_t target,
                                                     diopiTensorHandle_t weight, diopiTensorHandle_t buff,
                                                     diopiTensorHandle_t grad_input,
                                                     float gamma,
                                                     float alpha);

/**
 * @brief Uses iterative furthest point sampling to select a set of features whose corresponding points have the furthest distance.
 * @param[in] ctx diopi context.
 * @param points_xyz  (B, N, 3) where N > num_points.
 * @param num_points int: Number of points in the sampled set.
 * @param points_dist  (B, N, N) Distance between each point pair.
 * @param B, N points_xyz  (B, N, 3)
 * @param num_points int: Number of points in the sampled set.
 * @param[out] idx_tensor  (B, num_points) indices of the sampled points.
 * @param temp_tensor tmp result tensor for calculations.
 */
DIOPI_API diopiError_t diopiFurthestPointSampling(diopiContextHandle_t ctx, diopiConstTensorHandle_t points_xyz,
                                                  diopiTensorHandle_t temp_tensor, diopiTensorHandle_t idx_tensor,
                                                  int64_t B, int64_t N, int64_t num_points);
/**
 * @brief Uses iterative furthest point sampling to select a set of features whose corresponding points have the furthest distance(with dist version).
 * @param[in] ctx diopi context.
 * @param points_dist  (B, N, N) Distance between each point pair.
 * @sa definition of other parameters, refer to diopiFurthestPointSampling()
 */
DIOPI_API diopiError_t diopiFurthestPointSamplingWithDist(diopiContextHandle_t ctx, diopiConstTensorHandle_t points_dist,
                                                          diopiTensorHandle_t temp_tensor,
                                                          diopiTensorHandle_t idx_tensor, int64_t B,
                                                          int64_t N, int64_t num_points);

/**
 * @brief Calculate second order deviation.
 * @param[in] ctx diopi context.
 * @param input  Input feature map.
 * @param bias (nn.Parameter): The bias from convolution operation.
 * @param refer  empty tensor for calculations.
 * @param scale (float, optional): A scalar to adjust the variance of the feature
     map. Defaults to 2**0.5.
 * @param act =3
 * @param grad =0
 * @param alpha negative_slope (float, optional): Same as nn.LeakyRelu.
     Defaults to 0.2. alpha=negative_slope.
 * @param[out] out: Feature map after non-linear activation.
 */
DIOPI_API diopiError_t diopiFusedBiasLeakyrelu(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input,
                                               diopiConstTensorHandle_t bias, diopiConstTensorHandle_t refer, int64_t act, int64_t grad, float alpha, float scale);

/**
 * @brief Gather points with given index.
 * @param[in] ctx diopi context.
 * @param points features  (B, C, N) features to gather. points = features
 * @param idx indices (B, M) where M is the number of points. idx = indices
 * @param b,c,n points (b, c, n) features.
 * @param[out] out  (B, C, M) where M is the number of points.
 */
DIOPI_API diopiError_t diopiGatherPoints(diopiContextHandle_t ctx, diopiConstTensorHandle_t points, diopiConstTensorHandle_t idx, diopiTensorHandle_t out,
                                         int64_t b, int64_t c, int64_t n, int64_t npoints);
/**
 * @brief Backward function for Gather points with given index.
 * @param[in] ctx diopi context.
 * @param grad_out the gradient of ``out``.
 * @param[out] grad_points  the gradient of ``points``.
 * @sa definition of other parameters, refer to diopiGatherPoints()
 */
DIOPI_API diopiError_t diopiGatherPointsBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t idx, diopiTensorHandle_t grad_points,
                                                 int64_t b, int64_t c, int64_t n, int64_t npoints);

/**
 * @brief Groups points with a ball query of radius.
 * @param[in] ctx diopi context.
 * @param points features (Tensor): Tensor of features to group, input shape is
     (B, C, N) or stacked inputs (N1 + N2 ..., C). points=features/features_tensor.
 * @param idx indices (Tensor):  The indices of features to group with, input
     shape is (B, npoint, nsample) or stacked inputs
     (M1 + M2 ..., nsample). idx=indices.
 * @param b,c,n points (b, c, n)
 * @param npoints,nsample idx (B, npoint, nsample)
 * @param[out] out Grouped features, the shape is (B, C, npoint, nsample)
     or (M1 + M2 ..., C, nsample).
 */
DIOPI_API diopiError_t diopiGroupPoints(diopiContextHandle_t ctx, diopiConstTensorHandle_t points, diopiConstTensorHandle_t idx, diopiTensorHandle_t out,
                                        int64_t b, int64_t c, int64_t n, int64_t npoints, int64_t nsample);
/**
 * @brief Backward function for Groups points with a ball query of radius.
 * @param[in] ctx diopi context.
 * @param grad_out the gradient of ``out``.
 * @param[out] grad_points the gradient of ``points``.
 * @sa definition of other parameters, refer to diopiGroupPoints()
 */
DIOPI_API diopiError_t diopiGroupPointsBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t idx, diopiTensorHandle_t grad_points,
                                                int64_t b, int64_t c, int64_t n, int64_t npoints, int64_t nsample);

/**
 * @brief Groups points with a ball query of radius(stacked input).
 * @param[in] ctx diopi context.
 * @param features_tensor features (Tensor): Tensor of features to group, stacked input shape
      (N1 + N2 ..., C). features_tensor=features
 * @param idx_tensor indices (Tensor):  The indices of features to group with, stacked inputs
      (M1 + M2 ..., nsample). idx_tensor = indices.
 * @param features_batch_cnt_tensor (Tensor, optional): Input features nums in
       each batch, just like (N1, N2, ...). Defaults to None.
       New in version 1.7.0. .
 * @param idx_batch_cnt_tensor (Tensor, optional): Input indices nums in
       each batch, just like (M1, M2, ...). Defaults to None.
       New in version 1.7.0.
 * @param n,c features_tensor (n, c)
 * @param b idx_batch_cnt_tensor.shape[0]
 * @param nsample idx_tensor (m, nsample)
 * @param[out] out_tensor Grouped features, the shape is (B, C, npoint, nsample)
       or (M1 + M2 ..., C, nsample).
 */
DIOPI_API diopiError_t diopiStackGroupPoints(diopiContextHandle_t ctx,
                                             diopiConstTensorHandle_t features_tensor,
                                             diopiConstTensorHandle_t features_batch_cnt_tensor,
                                             diopiConstTensorHandle_t idx_tensor,
                                             diopiConstTensorHandle_t idx_batch_cnt_tensor,
                                             diopiTensorHandle_t out_tensor,
                                             int64_t b, int64_t c, int64_t m, int64_t nsample);
/**
 * @brief Backward function for Groups points with a ball query of radius(stacked input).
 * @param[in] ctx diopi context.
 * @param grad_out_tensor the gradient of ``out``.
 * @param[out] grad_features_tensor the gradient of ``features_tensor``.
 * @sa definition of other parameters, refer to diopiStackGroupPoints()
 */
DIOPI_API diopiError_t diopiStackGroupPointsBackward(diopiContextHandle_t ctx,
                                                     diopiConstTensorHandle_t grad_out_tensor,
                                                     diopiConstTensorHandle_t idx_tensor,
                                                     diopiConstTensorHandle_t idx_batch_cnt_tensor,
                                                     diopiConstTensorHandle_t features_batch_cnt_tensor,
                                                     diopiTensorHandle_t grad_features_tensor,
                                                     int64_t b, int64_t c, int64_t m, int64_t n, int64_t nsample);

/**
 * @brief Calculate boxes BEV overlap.
 * @param[in] ctx diopi context.
 * @param boxes_a: Input boxes a with shape (M, 7).
 * @param boxes_b: Input boxes b with shape (N, 7).
 * @param[out] ans_overlap: BEV overlap result with shape (M, N).
 */
DIOPI_API diopiError_t diopiIou3dBoxesOverlapBev(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes_a, diopiConstTensorHandle_t boxes_b, diopiTensorHandle_t ans_overlap);

/**
 * @brief 3D NMS function GPU implementation (for BEV boxes).
 * @param[in] ctx diopi context.
 * @param boxes  Input boxes with the shape of (N, 7)
     ([x, y, z, dx, dy, dz, heading]).
 * @param nms_overlap_thresh float: Overlap threshold of NMS.
 * @param[out] keep  Indexes after NMS.
 * @param keep_num  num_out for keep
 */
DIOPI_API diopiError_t diopiIou3dNms3d(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes, diopiTensorHandle_t keep, diopiTensorHandle_t keep_num, float nms_overlap_thresh);

/**
 * @brief Normal 3D NMS function GPU implementation. The overlap of two boxes for
          IoU calculation is defined as the exact overlapping area of the two boxes
          WITH their yaw angle set to 0.
 * @param[in] ctx diopi context.
 * @sa definition of other parameters, refer to diopiIou3dNms3d()
 */
DIOPI_API diopiError_t diopiIou3dNms3dNormal(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes, diopiTensorHandle_t keep, diopiTensorHandle_t keep_num, float nms_overlap_thresh);

/**
 * @brief KNN based on heap data structure.
 * @param[in] ctx diopi context.
 * @param xyz_tensor (B, N, 3) if transposed == False, else
     (B, 3, N). xyz coordinates of the features.
 * @param new_xyz_tensor   (B, npoint, 3) if transposed
     is False, else (B, 3, npoint). centers of the knn query.
     Default: None.
 * @param nsample int, number of nearest neighbors. 
 * @param b,n xyz_tensor (b, n, 3)
 * @param m = npoint
 * @param[out] idx_tensor  (B, npoint, k) tensor with the indices of the
     features that form k-nearest neighbours.
 * @param dist2_tensor  (B, npoint, k) distance tensors after calculations.
 */
DIOPI_API diopiError_t diopiKnn(diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz_tensor, diopiConstTensorHandle_t new_xyz_tensor, diopiTensorHandle_t idx_tensor,
                                diopiTensorHandle_t dist2_tensor, int64_t b, int64_t n, int64_t m, int64_t nsample);

/**
 * @brief A MaskedConv2d which inherits the official Conv2d. The masked forward doesn't implement the backward function and only
    supports the stride parameter to be 1 currently.
 */
DIOPI_API diopiError_t diopiMaskedIm2col(diopiContextHandle_t ctx, diopiConstTensorHandle_t im, diopiConstTensorHandle_t mask_h_idx,
                                         diopiConstTensorHandle_t mask_w_idx, diopiTensorHandle_t col,
                                         int64_t kernel_h, int64_t kernel_w,
                                         int64_t pad_h, int64_t pad_w);
DIOPI_API diopiError_t diopiMaskedCol2im(diopiContextHandle_t ctx, diopiConstTensorHandle_t col, diopiConstTensorHandle_t mask_h_idx,
                                         diopiConstTensorHandle_t mask_w_idx, diopiTensorHandle_t im, int64_t height,
                                         int64_t width, int64_t channels);

/**
 * @brief A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.
 */
DIOPI_API diopiError_t diopiModulatedDeformConv(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t weight, diopiTensorHandle_t bias, diopiTensorHandle_t ones, diopiTensorHandle_t offset,
                                                diopiTensorHandle_t mask, diopiTensorHandle_t output, diopiTensorHandle_t columns, int64_t kernel_h, int64_t kernel_w,
                                                const int64_t stride_h, const int64_t stride_w, const int64_t pad_h, const int64_t pad_w,
                                                const int64_t dilation_h, const int64_t dilation_w, const int64_t group,
                                                const int64_t deformable_group, const bool with_bias);
DIOPI_API diopiError_t diopiModulatedDeformConvBackward(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t weight, diopiTensorHandle_t bias, diopiTensorHandle_t ones, diopiTensorHandle_t offset,
                                                        diopiTensorHandle_t mask, diopiTensorHandle_t columns, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                        diopiTensorHandle_t grad_bias, diopiTensorHandle_t grad_offset, diopiTensorHandle_t grad_mask, diopiTensorHandle_t grad_output,
                                                        int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w, int64_t pad_h,
                                                        int64_t pad_w, int64_t dilation_h, int64_t dilation_w, int64_t group, int64_t deformable_group,
                                                        const bool with_bias);

/**
 * @brief An attention module used in Deformable-Detr.
 * @param[in] ctx diopi context.
 * @param value   The value has shape
     (bs, num_keys, mum_heads, embed_dims//num_heads)
 * @param spatial_shapes  Spatial shape of
     each feature map, has shape (num_levels, 2),
     last dimension 2 represent (h, w). 
 * @param level_start_index: level_start_index input tensor.
 * @param sampling_loc   The location of sampling points,
     has shape (bs ,num_queries, num_heads, num_levels, num_points, 2),
     the last dimension 2 represent (x, y). 
 * @param attn_weight  The weight of sampling points
     used when calculate the attention, has shape
     (bs ,num_queries, num_heads, num_levels, num_points).
 * @param im2col_step   The step used in image to column.
 * @param[out] out  attention module out which has shape (bs, num_queries, embed_dims)
 */
DIOPI_API diopiError_t diopiMsDeformAttn(diopiContextHandle_t ctx, diopiTensorHandle_t* out,
                                         diopiConstTensorHandle_t value,
                                         diopiConstTensorHandle_t spatial_shapes,
                                         diopiConstTensorHandle_t level_start_index,
                                         diopiConstTensorHandle_t sampling_loc,
                                         diopiConstTensorHandle_t attn_weight,
                                         int64_t im2col_step);
/**
 * @brief Backward function for An attention module used in Deformable-Detr.
 * @param[in] ctx diopi context.
 * @param grad_output  the gradient of ``out``.
 * @param[out] grad_value,grad_sampling_loc,grad_attn_weight  the gradient of ``value``,``sampling_loc``,``attn_weight``.
 * @sa definition of other parameters, refer to diopiMsDeformAttn()
 */
DIOPI_API diopiError_t diopiMsDeformAttnBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t value, diopiConstTensorHandle_t spatial_shapes,
                                                 diopiConstTensorHandle_t level_start_index, diopiConstTensorHandle_t sampling_loc,
                                                 diopiConstTensorHandle_t attn_weight, diopiConstTensorHandle_t grad_output, diopiTensorHandle_t grad_value,
                                                 diopiTensorHandle_t grad_sampling_loc, diopiTensorHandle_t grad_attn_weight, int64_t im2col_step);

/**
 * @brief NMS from mmcv. This function is modified from: https://github.com/pytorch/vision/
 * @param[in] ctx diopi context.
 * @param dets boxes boxes in shape (N, 4). dets=boxes.
 * @param scores scores in shape (N, ).
 * @param iou_threshold float: IoU threshold for NMS.
 * @param offset (int, 0 or 1): boxes' width or height is (x2 - x1 + offset).
 * @param[out] out indice, which always have the same data type as the input.
 */
DIOPI_API diopiError_t diopiNmsMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t dets,
                                    diopiConstTensorHandle_t scores, double iou_threshold, int64_t offset);

/**
 * @brief Performs non-maximum suppression (NMS) on the rotated boxes according to their intersection-over-union (IoU).
 * @param[in] ctx diopi context.
 * @param dets Rotated boxes in shape (N, 5). They are expected to be in
     (x_ctr, y_ctr, width, height, angle_radian) format.
 * @param order_t, dets_sorted order tensor and ordered dets
 * @param scores  scores in shape (N, ).
 * @param iou_threshold float: IoU thresh for NMS.
 * @param multi_label  boxes' label in shape (N,).
 * @param out indice, which is always the same data type as the input.
 */
DIOPI_API diopiError_t diopiNmsRotated(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t dets,
                                       diopiConstTensorHandle_t scores, diopiConstTensorHandle_t order_t, diopiConstTensorHandle_t dets_sorted,
                                       double iou_threshold, int64_t multi_label);

/**
 * @brief Find the box in which each point is(Part).
 * @param[in] ctx diopi context.
 * @param pts points [B, M, 3], [x, y, z] in LiDAR/DEPTH coordinate. pts=points
 * @param boxes   [B, T, 7],
     num_valid_boxes <= T, [x, y, z, x_size, y_size, z_size, rz] in
     LiDAR/DEPTH coordinate, (x, y, z) is the bottom center.
 * @param[out] box_idx_of_points  Return the box indices of points with the shape of
     (B, M). Default background = -1.
 */
DIOPI_API diopiError_t diopiPointsInBoxesPart(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes, diopiConstTensorHandle_t pts, diopiTensorHandle_t box_idx_of_points);
/**
 * @brief Find the box in which each point is(All).
 * @param[in] ctx diopi context.
 * @param pts points [B, M, 3], [x, y, z] in LiDAR/DEPTH coordinate. pts=points
 * @param boxes   [B, T, 7],
     num_valid_boxes <= T, [x, y, z, x_size, y_size, z_size, rz] in
     LiDAR/DEPTH coordinate, (x, y, z) is the bottom center.
 * @param[out] box_idx_of_points  Return the box indices of points with the shape of
     (B, M). Default background = -1.
 */
DIOPI_API diopiError_t diopiPointsInBoxesAll(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes, diopiConstTensorHandle_t pts, diopiTensorHandle_t box_idx_of_points);

/**
 * @brief Psamask. Modified from https://github.com/hszhao/semseg/blob/master/lib/psa
 */
DIOPI_API diopiError_t diopiPsamask(diopiContextHandle_t ctx, int64_t psa_type, diopiConstTensorHandle_t input, diopiTensorHandle_t output,
                                    int64_t num_, int64_t h_feature,
                                    int64_t w_feature, int64_t h_mask,
                                    int64_t w_mask, int64_t half_h_mask,
                                    int64_t half_w_mask);
DIOPI_API diopiError_t diopiPsamaskBackward(diopiContextHandle_t ctx, int64_t psa_type, diopiConstTensorHandle_t grad_output,
                                            diopiTensorHandle_t grad_input, int64_t num_,
                                            int64_t h_feature, int64_t w_feature,
                                            int64_t h_mask, int64_t w_mask,
                                            int64_t half_h_mask, int64_t half_w_mask);

/**
 * @brief RoI align pooling layer.
 */
DIOPI_API diopiError_t diopiRoiAlignMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t rois, diopiTensorHandle_t output,
                                         diopiTensorHandle_t argmax_y, diopiTensorHandle_t argmax_x,
                                         int64_t aligned_height, int64_t aligned_width,
                                         float spatial_scale, int64_t sampling_ratio,
                                         int64_t pool_mode, bool aligned);
DIOPI_API diopiError_t diopiRoiAlignBackwardMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiTensorHandle_t rois, diopiTensorHandle_t argmax_y,
                                                 diopiTensorHandle_t argmax_x, diopiTensorHandle_t grad_input,
                                                 int64_t aligned_height, int64_t aligned_width,
                                                 float spatial_scale, int64_t sampling_ratio,
                                                 int64_t pool_mode, bool aligned);

/**
 * @brief RoI align pooling layer for rotated proposals for mmcv.
 * @param[in] ctx diopi context.
 * @param input  a feature map of shape (N, C, H, W)
 * @param rois  rois with shape (n, 6) with each roi decoded as (batch_index, center_x, center_y,
     w, h, angle). The angle is in radian.
 * @param aligned_height,aligned_width  output_size (tuple): h, w. aligned_height=output_size[0], aligned_width=output_size[1]
 * @param spatial_scale float: scale the input boxes by this number
 * @param sampling_ratio int  number of inputs samples to take for each
     output sample. 0 to take samples densely for current models.
 * @param aligned (bool): if False, use the legacy implementation in
     MMDetection. If True, align the results more perfectly.
     Default: True.
 * @param clockwise (bool): If True, the angle in each proposal follows a
     clockwise fashion in image space, otherwise, the angle is
     counterclockwise. Default: False.
 * @param[out] output  RoI align pooling layer output.
 */
DIOPI_API diopiError_t diopiRoiAlignRotated(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t rois, diopiTensorHandle_t output,
                                            int64_t aligned_height, int64_t aligned_width,
                                            float spatial_scale, int64_t sampling_ratio,
                                            bool aligned, bool clockwise);
/**
 * @brief Backward function for RoI align pooling layer for rotated proposals for mmcv.
 * @param[in] ctx diopi context.
 * @param top_grad   the gradient of ``output``.
 * @param[out] bottom_grad   the gradient of ``input``.
 * @sa definition of other parameters, refer to diopiRoiAlignRotated()
 */
DIOPI_API diopiError_t diopiRoiAlignRotatedBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t top_grad, diopiConstTensorHandle_t rois,
                                                    diopiTensorHandle_t bottom_grad, int64_t aligned_height,
                                                    int64_t aligned_width, float spatial_scale,
                                                    int64_t sampling_ratio, bool aligned,
                                                    bool clockwise);

/**
 * @brief Rotation-invariant RoI align pooling layer for rotated proposals.
 * @param[in] ctx diopi context.
 * @param features input a feature map of shape (N, C, H, W) features=input
 * @param rois rois with shape (n, 6) with each roi decoded as (batch_index, center_x, center_y,
     w, h, angle). The angle is in radian.
 * @param pooled_height,pooled_width out_size (tuple): fixed dimensional RoI output with shape (h, w). pooled_height=h, pooled_width=w
 * @param spatial_scale float: scale the input boxes by this number
 * @param num_samples int: number of inputs samples to take for each
     output sample. 0 to take samples densely for current models.
 * @param num_orientations int: number of oriented channels.
 * @param clockwise (bool): If True, the angle in each proposal follows a
     clockwise fashion in image space, otherwise, the angle is
     counterclockwise. Default: False.
 * @param[out] output Rotation-invariant RoI align pooling layer output.
 */
DIOPI_API diopiError_t diopiRiroiAlignRotated(diopiContextHandle_t ctx, diopiConstTensorHandle_t features, diopiConstTensorHandle_t rois,
                                              diopiTensorHandle_t output, int64_t pooled_height,
                                              int64_t pooled_width, float spatial_scale,
                                              int64_t num_samples, int64_t num_orientations,
                                              bool clockwise);
/**
 * @brief Backward function for Rotation-invariant RoI align pooling layer for rotated proposals.
 * @param[in] ctx diopi context.
 * @param top_grad   the gradient of ``output``.
 * @param[out] bottom_grad   the gradient of ``features``.
 * @sa definition of other parameters, refer to diopiRiroiAlignRotated()
 */
DIOPI_API diopiError_t diopiRiroiAlignRotatedBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t top_grad, diopiConstTensorHandle_t rois,
                                                      diopiTensorHandle_t bottom_grad, int64_t pooled_height,
                                                      int64_t pooled_width, float spatial_scale,
                                                      int64_t num_samples, int64_t num_orientations,
                                                      bool clockwise);

/**
 * @brief Encode the geometry-specific features of each 3D proposal.
 * @param[in] ctx diopi context.
 * @param rois : [N, 7], in LiDAR coordinate,
         (x, y, z) is the bottom center of rois.
 * @param pts : [npoints, 3], coordinates of input points.
 * @param pts_feature : [npoints, C], features of input points.
 * @param pool_method int: Pooling method of RoIAware, 0 (max pool) or 1 (average
     pool).
 * @param[out] pooled_features: Pooled features whose shape is [N, out_x, out_y, out_z, C].
 * @param argmax, pts_idx_of_voxels: used in backward function.
 */
DIOPI_API diopiError_t diopiRoiawarePool3d(diopiContextHandle_t ctx, diopiConstTensorHandle_t rois,
                                           diopiConstTensorHandle_t pts, diopiConstTensorHandle_t pts_feature,
                                           diopiTensorHandle_t argmax, diopiTensorHandle_t pts_idx_of_voxels,
                                           diopiTensorHandle_t pooled_features, int64_t pool_method);
/**
 * @brief Backward function for Encode the geometry-specific features of each 3D proposal.
 * @param[in] ctx diopi context.
 * @param grad_out   the gradient of ``pooled_features``.
 * @param[out] grad_in   the gradient of ``pts_feature``.
 * @sa definition of other parameters, refer to diopiRoiawarePool3d()
 */
DIOPI_API diopiError_t diopiRoiawarePool3dBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t pts_idx_of_voxels,
                                                   diopiConstTensorHandle_t argmax, diopiConstTensorHandle_t grad_out,
                                                   diopiTensorHandle_t grad_in, int64_t pool_method);

/**
 * @brief Encode the geometry-specific features of each 3D proposal.
 * @param[in] ctx diopi context.
 * @param xyz   Input points whose shape is (B, N, C). 
 * @param pts_feature  Features of input points whose shape
      is (B, N, C). 
 * @param boxes3d (B, M, 7), Input bounding boxes whose shape is (B, M, 7).
 * @param[out] pooled_features, pooled_empty_flag A tuple contains two elements. The first one
      is the pooled features whose shape is (B, M, 512, 3 + C). The
      second is an empty flag whose shape is (B, M).
 */
DIOPI_API diopiError_t diopiRoipointPool3d(diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz, diopiConstTensorHandle_t boxes3d,
                                           diopiConstTensorHandle_t pts_feature, diopiTensorHandle_t pooled_features, diopiTensorHandle_t pooled_empty_flag);

/**
 * @brief RoiPool.
 */
DIOPI_API diopiError_t diopiRoiPool(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t rois, diopiTensorHandle_t output,
                                    diopiTensorHandle_t argmax, int64_t pooled_height, int64_t pooled_width,
                                    float spatial_scale);
DIOPI_API diopiError_t diopiRoiPoolBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiTensorHandle_t rois, diopiTensorHandle_t argmax,
                                            diopiTensorHandle_t grad_input, int64_t pooled_height,
                                            int64_t pooled_width, float spatial_scale);

/**
 * @brief Scatters points into voxels, used in the voxel encoder with dynamic voxelization.
 */
DIOPI_API diopiError_t diopiDynamicPointToVoxel(diopiContextHandle_t ctx, diopiTensorHandle_t* outlist, diopiConstTensorHandle_t feats, diopiConstTensorHandle_t coors,
                                                int64_t reduce_type);
DIOPI_API diopiError_t diopiDynamicPointToVoxelBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_feats, diopiConstTensorHandle_t grad_reduced_feats,
                                                        diopiConstTensorHandle_t feats, diopiConstTensorHandle_t reduced_feats,
                                                        diopiConstTensorHandle_t coors_idx, diopiConstTensorHandle_t reduce_count,
                                                        int64_t reduce_type);

/**
 * @brief Synchronized Batch Normalization.
 * @param[in] ctx diopi context.
 * @param input input feature.
 * @param[out] mean initialized as zeros tensors. mean feature tensor calculated from diopiSyncBnMean().
 */
DIOPI_API diopiError_t diopiSyncBnMean(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiTensorHandle_t mean);
/**
 * @brief Synchronized Batch Normalization.
 * @param[in] ctx diopi context.
 * @param input,mean  input featire.
 * @param[out] var initialized as zeros tensors. var feature tensor calculated from diopiSyncBnVar() after diopiSyncBnMean().
 * @sa definition of other parameters, refer to diopiSyncBnMean()
 */
DIOPI_API diopiError_t diopiSyncBnVar(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean,
                                      diopiTensorHandle_t var);
/**
 * @brief Synchronized Batch Normalization.
 * @param[in] ctx diopi context.
 * @param input,weight,bias  input featire.
 * @param eps (float, optional): a value added to the denominator for numerical
     stability. Defaults to 1e-5.
 * @param momentum (float, optional): the value used for the running_mean and
     running_var computation. Defaults to 0.1.
 * @param group_size  group_size.
 * @param[out] norm,std  initialized as zeros tensors. feature tensor calculated from diopiSyncBnOutput() after diopiSyncBnMean() and iopiSyncBnVar().
 * @param running_mean,running_var calculated mean and var during forward function.
 * @param output  Synchronized Batch Normalization output tensor.
 * @sa definition of other parameters, refer to diopiSyncBnMean(), diopiSyncBnMean().
 */
DIOPI_API diopiError_t diopiSyncBnOutput(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean,
                                         diopiConstTensorHandle_t var, diopiTensorHandle_t running_mean,
                                         diopiTensorHandle_t running_var, diopiConstTensorHandle_t weight,
                                         diopiConstTensorHandle_t bias, diopiTensorHandle_t norm, diopiTensorHandle_t std,
                                         diopiTensorHandle_t output, float eps, float momentum, int64_t group_size);
/**
 * @brief Backward function for Synchronized Batch Normalization(grad for param).
 * @param[in] ctx diopi context.
 * @param grad_output  the gradient of ``output``.
 * @param[out] grad_weight the gradient of ``weight``.
 * @param grad_bias  the gradient of ``bias``.
 * @sa definition of other parameters, refer to diopiSyncBnOutput().
 */
DIOPI_API diopiError_t diopiSyncBnBackwardParam(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t norm,
                                                diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias);
/**
 * @brief Backward function for Synchronized Batch Normalization(grad for input).
 * @param[in] ctx diopi context.
 * @param[out] grad_input the gradient of ``input``.
 * @sa definition of other parameters, refer to diopiSyncBnMean(), diopiSyncBnOutput().
 */
DIOPI_API diopiError_t diopiSyncBnBackwardData(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t weight,
                                               diopiConstTensorHandle_t grad_weight, diopiConstTensorHandle_t grad_bias, diopiConstTensorHandle_t norm,
                                               diopiConstTensorHandle_t std, diopiTensorHandle_t grad_input);

/**
 * @brief Performs weighted linear interpolation on 3 features.
 * @param[in] ctx diopi context.
 * @param points   (B, C, M) Features descriptors to be
     interpolated. 
 * @param idx   (B, n, 3) indices of three nearest
     neighbor features for the target features. 
 * @param weight  (B, n, 3) weights of three nearest
     neighbor features for the target features.
 * @param b,c,n,m  points   (b, c, m)  idx  (b, n, 3) 
 * @param[out] out (B, C, N) tensor of the interpolated features
 */
DIOPI_API diopiError_t diopiThreeInterpolate(diopiContextHandle_t ctx, diopiConstTensorHandle_t points, diopiConstTensorHandle_t idx,
                                             diopiConstTensorHandle_t weight, diopiTensorHandle_t out, int64_t b, int64_t c, int64_t m, int64_t n);
/**
 * @brief Backward function for Performs weighted linear interpolation on 3 features.
 * @param[in] ctx diopi context.
 * @param grad_out   the gradient of ``out``.
 * @param[out] grad_points the gradient of ``points``.
 * @sa definition of other parameters, refer to diopiThreeInterpolate().
 */
DIOPI_API diopiError_t diopiThreeInterpolateBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t idx,
                                                     diopiConstTensorHandle_t weight, diopiTensorHandle_t grad_points, int64_t b, int64_t c, int64_t n, int64_t m);

/**
 * @brief Find the top-3 nearest neighbors of the target set from the source set.
 * @param[in] ctx diopi context.
 * @param unknown   shape (B, N, 3), points set that needs to
     find the nearest neighbors. 
 * @param known   shape (B, M, 3), points set that is used
     to find the nearest neighbors of points in target set. 
 * @param b,n,m  unknown : shape (b, n, 3) known : shape (b, m, 3)
 * @param[out] dist2  shape (B, N, 3), L2 distance of each point in target set to their corresponding top three nearest neighbors.
 * @param idx  index tensor for reference.
 */
DIOPI_API diopiError_t diopiThreeNn(diopiContextHandle_t ctx, diopiConstTensorHandle_t unknown, diopiConstTensorHandle_t known, diopiTensorHandle_t dist2,
                                    diopiTensorHandle_t idx, int64_t b, int64_t n, int64_t m);

/**
 * @brief Temporal Interlace Shift.
 * @param[in] ctx diopi context.
 * @param input Feature map with shape [N, num_segments, C, H * W].
 * @param shift   Shift tensor with shape [N, num_segments].
 * @param[out] output Feature map after temporal interlace shift.
 */
DIOPI_API diopiError_t diopiTinShift(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t shift, diopiTensorHandle_t output);
/**
 * @brief Backward function for Temporal Interlace Shift.
 * @param[in] ctx diopi context.
 * @param grad_output   the gradient of ``output``.
 * @param[out] grad_input the gradient of ``input``.
 * @sa definition of other parameters, refer to diopiTinShift().
 */
DIOPI_API diopiError_t diopiTinShiftBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t shift, diopiTensorHandle_t grad_input);

/**
 * @brief UpFRIDn for 2d features.
 * @param[in] ctx diopi context.
 * @param input  Tensor with shape of (n, c, h, w).
 * @param kernel   Filter kernel.
 * @param up_x,up_y up (int | tuple[int], optional): Upsampling factor. If given a number,
     we will use this factor for the both height and width side.
     Defaults to 1. up_x, up_y = up
 * @param down_x,down_y down (int | tuple[int], optional): Downsampling factor. If given a
     number, we will use this factor for the both height and width side.
     Defaults to 1. down_x, down_y = down
 * @param pad_x0, pad_x1, pad_y0, pad_y1 pad (tuple[int], optional): Padding for tensors, (x_pad, y_pad) or
     (x_pad_0, x_pad_1, y_pad_0, y_pad_1). Defaults to (0, 0). 
     pad_x0, pad_x1, pad_y0, pad_y1 = pad
 * @param[out] out  Tensor after UpFIRDn.
 */
DIOPI_API diopiError_t diopiUpfirdn2dOp(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input,
                                        diopiConstTensorHandle_t kernel, int64_t up_x, int64_t up_y,
                                        int64_t down_x, int64_t down_y, int64_t pad_x0, int64_t pad_x1,
                                        int64_t pad_y0, int64_t pad_y1);

/**
 * @brief Convert kitti points(N, >=3) to voxels.
 *  @param[in] ctx diopi context.
 * @param points  [N, ndim]. Points[:, :3] contain xyz points
     and points[:, 3:] contain other information like reflectivity.
 * @param voxel_size (tuple or float): The size of voxel with the shape of
     [3].
 * @param coors_range (tuple or float): The coordinate range of voxel with
     the shape of [6].
 * @param max_points (int, optional): maximum points contained in a voxel. if
     max_points=-1, it means using dynamic_voxelize. Default: 35.
 * @param max_voxels (int, optional): maximum voxels this function create.
     for second, 20000 is a good choice. Users should shuffle points
     before call this function because max_voxels may drop points.
     Default: 20000.
 * @param NDim =3.
 * @param deterministic  bool. whether to invoke the non-deterministic
     version of hard-voxelization implementations. non-deterministic
     version is considerablly fast but is not deterministic. only
     affects hard voxelization. default True. for more information
     of this argument and the implementation insights, please refer
     to the following links:
     https://github.com/open-mmlab/mmdetection3d/issues/894
     https://github.com/open-mmlab/mmdetection3d/pull/904
     it is an experimental feature and we will appreciate it if
     you could share with us the failing cases.
 * @param[out] voxels, coors, num_points_per_voxel, voxel_num A tuple contains three elements. The first one is the output voxels with the shape of
     [M, max_points, n_dim], which only contain points and returned when max_points != -1.
     The second is the voxel coordinates with shape of [M, 3]. The last is number of point per voxel with the
     shape of [M], which only returned when max_points != -1. voxel_num is for index select.
 */
DIOPI_API diopiError_t diopiHardVoxelize(diopiContextHandle_t ctx, diopiConstTensorHandle_t points,
                                         diopiConstTensorHandle_t voxel_size,
                                         diopiConstTensorHandle_t coors_range, diopiTensorHandle_t voxels,
                                         diopiTensorHandle_t coors, diopiTensorHandle_t num_points_per_voxel,
                                         diopiTensorHandle_t voxel_num, const int64_t max_points,
                                         const int64_t max_voxels, const int64_t NDim,
                                         const bool deterministic);
/**
 * @brief Convert kitti points(N, >=3) to voxels(max_points == -1 or max_voxels == -1).
 *  @param[in] ctx diopi context.
 * @sa definition of other parameters, refer to diopiHardVoxelize().
 */
DIOPI_API diopiError_t diopiDynamicVoxelize(diopiContextHandle_t ctx, diopiConstTensorHandle_t points,
                                            diopiConstTensorHandle_t voxel_size,
                                            diopiConstTensorHandle_t coors_range, diopiTensorHandle_t coors,
                                            const int64_t NDim);

/**
 * @brief Using the feature interpolation to obtain the position information
          correspond to the refined rotate anchors and reconstruct the feature maps
          in pixel-wise manner to achieve feature alignment.
 * @param[in] ctx diopi context.
 * @param features   Input features with shape [N,C,H,W].
 * @param best_bboxes  Refined rotate anchors with
     shape [N,H,W,5]. Coordinate format (cx,cx,h,w,a). 
 * @param spatial_scale float: The scale of feature map size and
     input image size.
 * @param points (int, optional): The number of sample points.
     Only 1 and 5 are supported. Defaults to 1.
 * @param[out] output  Refined features with shape [N,C,H,W].
 */
DIOPI_API diopiError_t diopiRotatedFeatureAlign(diopiContextHandle_t ctx, diopiConstTensorHandle_t features,
                                                diopiConstTensorHandle_t best_bboxes,
                                                float spatial_scale,
                                                int64_t points, diopiTensorHandle_t output);
/**
 * @brief Backward function for Using the feature interpolation to obtain the position information
          correspond to the refined rotate anchors and reconstruct the feature maps
          in pixel-wise manner to achieve feature alignment.
 * @param[in] ctx diopi context.
 * @param top_grad   the gradient of ``output``.
 * @param[out] bottom_grad the gradient of ``features``.
 * @sa definition of other parameters, refer to diopiRotatedFeatureAlign().
 */
DIOPI_API diopiError_t diopiRotatedFeatureAlignBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t top_grad,
                                                        diopiConstTensorHandle_t best_bboxes,
                                                        float spatial_scale,
                                                        int64_t points, diopiTensorHandle_t bottom_grad);

/**
 * @brief Judging whether points are inside polygons, which is used in the ATSS assignment for the rotated boxes.
 * @param[in] ctx diopi context.
 * @param points  It has shape (B, 2), indicating (x, y).
     M means the number of predicted points.
 * @param polygons  It has shape (M, 8), indicating
     (x1, y1, x2, y2, x3, y3, x4, y4). M means the number of
     ground truth polygons.
 * @param[out] output  Return the result with the shape of (B, M),
     1 indicates that the point is inside the polygon,
     0 indicates that the point is outside the polygon.
 */
DIOPI_API diopiError_t diopiPointsInPolygons(diopiContextHandle_t ctx, diopiConstTensorHandle_t points, diopiConstTensorHandle_t polygons, diopiTensorHandle_t output);

/**
 * @brief Sparse ops.(only support pytorch now)
 */
DIOPI_API diopiError_t diopiIndiceMaxpool(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiTensorHandle_t features,
                                          diopiTensorHandle_t indicePairs,
                                          diopiTensorHandle_t indiceNum,
                                          int64_t numAct);
DIOPI_API diopiError_t diopiIndiceMaxpoolBackward(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiTensorHandle_t features,
                                                  diopiTensorHandle_t outFeatures,
                                                  diopiTensorHandle_t outGrad,
                                                  diopiTensorHandle_t indicePairs,
                                                  diopiTensorHandle_t indiceNum);
DIOPI_API diopiError_t diopiIndiceConv(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiTensorHandle_t features,
                                       diopiTensorHandle_t filters,
                                       diopiTensorHandle_t indicePairs,
                                       diopiTensorHandle_t indiceNum,
                                       int64_t numActOut, int64_t _inverse,
                                       int64_t _subM);
DIOPI_API diopiError_t diopiIndiceConvBackward(diopiContextHandle_t ctx, diopiTensorHandle_t* outlist,
                                               diopiTensorHandle_t features, diopiTensorHandle_t filters, diopiTensorHandle_t outGrad,
                                               diopiTensorHandle_t indicePairs, diopiTensorHandle_t indiceNum, int64_t _inverse,
                                               int64_t _subM);
DIOPI_API diopiError_t diopiFusedIndiceConvBatchnorm(diopiContextHandle_t ctx, diopiTensorHandle_t* out,
                                                     diopiTensorHandle_t features, diopiTensorHandle_t filters, diopiTensorHandle_t bias,
                                                     diopiTensorHandle_t indicePairs, diopiTensorHandle_t indiceNum, int64_t numActOut,
                                                     int64_t _inverse, int64_t _subM);

/**
 * @brief Find the smallest polygons that surrounds all points in the point sets.
 * @param[in] ctx diopi context.
 * @param pointsets  point sets with shape  (N, 18).
 * @param[out] polygons Return the smallest polygons with shape (N, 8).
 */
DIOPI_API diopiError_t diopiMinAreaPolygons(diopiContextHandle_t ctx, diopiConstTensorHandle_t pointsets, diopiTensorHandle_t polygons);

/**
 * @brief Encoding the orientation information and generating orientation-sensitive features.
 * @param[in] ctx diopi context.
 * @param input  Input features with shape
     [num_output_planes, num_input_planes, num_orientations, H, W].
 * @param indices  Indices with shape
     [num_orientations, H, W, num_rotations].
 * @param[out] output   Refined features with shape [num_output_planes *
     num_rotations, num_input_planes * num_orientations, H, W].
 */
DIOPI_API diopiError_t diopiActiveRotatedFilter(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices, diopiTensorHandle_t output);
/**
 * @brief  Backward function for Encoding the orientation information and generating orientation-sensitive features.
 * @param[in] ctx diopi context.
 * @param grad_out   the gradient of ``output``.
 * @param[out] grad_in the gradient of ``input``.
 * @sa definition of other parameters, refer to diopiActiveRotatedFilter().
 */
DIOPI_API diopiError_t diopiActiveRotatedFilterBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t indices, diopiTensorHandle_t grad_in);

/**
 * @brief Return generalized intersection-over-union (Jaccard index) between point sets and polygons(ConvexIou version).
 * @param[in] ctx diopi context.
 * @param pointsets  It has shape (N, 18),
     indicating (x1, y1, x2, y2, ..., x9, y9) for each row.
 * @param polygons  It has shape (N, 8),
     indicating (x1, y1, x2, y2, x3, y3, x4, y4) for each row.
 * @param[out] ious  Return the ious between point sets and polygons with the
     shape (N, K).
 */
DIOPI_API diopiError_t diopiConvexIou(diopiContextHandle_t ctx, diopiConstTensorHandle_t pointsets, diopiConstTensorHandle_t polygons, diopiTensorHandle_t ious);
/**
 * @brief Return generalized intersection-over-union (Jaccard index) between point sets and polygons(ConvexGiou version).
 * @param[in] ctx diopi context.
 * @param[out] output  The first element is the gious
     between point sets and polygons with the shape (N,). The second
     element is the gradient of point sets with the shape (N, 18).
 * @sa definition of other parameters, refer to diopiConvexIou().
 */
DIOPI_API diopiError_t diopiConvexGiou(diopiContextHandle_t ctx, diopiConstTensorHandle_t pointsets, diopiConstTensorHandle_t polygons, diopiTensorHandle_t output);

/**
 * @brief Sort indices.
    Note:
        why 9? the polygon has maximal 8 vertices.
        +1 to duplicate the first element.
        the index should have following structure:
            (A, B, C, ... , A, X, X, X)
        and X indicates the index of arbitrary elements in the last
        16 (intersections not corners) with value 0 and mask False.
        (cause they have zero value and zero gradient)
 * @param[in] ctx diopi context.
 * @param vertices  (B, N, 24, 2) Box vertices.
 * @param mask  (B, N, 24) Mask.
 * @param num_valid  (B, N) sum of mask, dim=2.
 * @param[out] out (B, N, 9) Sorted indices.
 */
DIOPI_API diopiError_t diopiDiffIouRotatedSortVertices(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t vertices, diopiConstTensorHandle_t mask,
                                                       diopiConstTensorHandle_t num_valid);

/**
 * @brief This is an implementation of the 2D Chamfer Distance. It has been used in the paper `Oriented RepPoints for Aerial Object
 *        Detection (CVPR 2022) <https://arxiv.org/abs/2105.11111>_`.
 * @param[in] ctx diopi context.
 * @param xyz1 (Tensor): Point set with shape (B, N, 2). 
 * @param xyz2 (Tensor): Point set with shape (B, N, 2). 
 * @param[out] dist1,dist2,idx1,idx2 out sequence[Tensor]:
     - dist1 (Tensor): Chamfer distance (xyz1 to xyz2) with
         shape (B, N). 
     - dist2 (Tensor): Chamfer distance (xyz2 to xyz1) with
         shape (B, N). 
     - idx1 (Tensor): Index of chamfer distance (xyz1 to xyz2)
         with shape (B, N), which be used in compute gradient. 
     - idx2 (Tensor): Index of chamfer distance (xyz2 to xyz2)
         with shape (B, N), which be used in compute gradient. 
 */
DIOPI_API diopiError_t diopiChamferDistance(diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz1, diopiConstTensorHandle_t xyz2, diopiTensorHandle_t dist1,
                                            diopiTensorHandle_t dist2, diopiTensorHandle_t idx1, diopiTensorHandle_t idx2);
/**
 * @brief Backward function for the 2D Chamfer Distance. It has been used in the paper `Oriented RepPoints for Aerial Object
 *        Detection (CVPR 2022) <https://arxiv.org/abs/2105.11111>_`.
 * @param[in] ctx diopi context.
 * @param grad_dist1   the gradient of ``dist1``.
 * @param grad_dist2   the gradient of ``dist2``.
 * @param[out] grad_xyz1 the gradient of ``xyz1``.
 * @param grad_xyz2   the gradient of ``xyz2``.
 * @sa definition of other parameters, refer to diopiChamferDistance().
 */
DIOPI_API diopiError_t diopiChamferDistanceBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz1, diopiConstTensorHandle_t xyz2,
                                                    diopiConstTensorHandle_t idx1, diopiConstTensorHandle_t idx2, diopiConstTensorHandle_t grad_dist1, diopiConstTensorHandle_t grad_dist2,
                                                    diopiTensorHandle_t grad_xyz1, diopiTensorHandle_t grad_xyz2);

/**
 * @brief The operation of precision RoI pooling. The implementation of PrRoIPool is modified from https://github.com/vacancy/PreciseRoIPooling/
 * @param[in] ctx diopi context.
 * @param input (torch.Tensor): The feature map. 
 * @param rois (torch.Tensor): The RoI bboxes in [tl_x, tl_y, br_x, br_y]
     format.
 * @param pooled_height,pooled_width output_size (Union[int, tuple]): h, w. pooled_height=h pooled_width=w.
 * @param spatial_scale (float, optional): scale the input boxes by this number.
     Defaults to 1.0.
 * @param[out] output The pooled results.
 */
DIOPI_API diopiError_t diopiPrroiPool(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t rois, diopiTensorHandle_t output,
                                      int64_t pooled_height, int64_t pooled_width, float spatial_scale);
/**
 * @brief Backward function for the operation of precision RoI pooling. The implementation of PrRoIPool is modified from https://github.com/vacancy/PreciseRoIPooling/(grad of input)
 * @param[in] ctx diopi context.
 * @param grad_output   the gradient of ``output``.
 * @param[out] grad_input the gradient of ``input``.
 * @sa definition of other parameters, refer to diopiPrroiPool().
 */
DIOPI_API diopiError_t diopiPrroiPoolbackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t rois, diopiTensorHandle_t grad_input,
                                              int64_t pooled_height, int64_t pooled_width, float spatial_scale);
/**
 * @brief The operation of precision RoI pooling. The implementation of PrRoIPool is modified from https://github.com/vacancy/PreciseRoIPooling/(grad of rois)
 * @param[in] ctx diopi context.
 * @param[out] grad_rois the gradient of ``rois``.
 * @sa definition of other parameters, refer to diopiPrroiPool(), diopiPrroiPoolbackward().
 */
DIOPI_API diopiError_t diopiPrroiPoolCoorBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t output, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t rois, diopiTensorHandle_t grad_rois,
                                                  int64_t pooled_height, int64_t pooled_width, float spatial_scale);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_H_
