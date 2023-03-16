/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/

#ifndef _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_MMCV_H_
#define _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_MMCV_H_

#include <diopi/diopirt.h>

#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus

/* DIOPI functions from MMCV extension ops <https://github.com/open-mmlab/mmcv.git>*/

/**
 * \brief Perform weighted sum to generate output features according to scores.
 */
DIOPI_API diopiError_t diopiAssignScoreWithk(diopiContextHandle_t ctx, diopiConstTensorHandle_t points, diopiConstTensorHandle_t centers,
                                diopiConstTensorHandle_t scores, diopiConstTensorHandle_t knn_idx,
                                diopiTensorHandle_t output, int64_t B, int64_t N0, int64_t N1, int64_t M,
                                int64_t K, int64_t O, int64_t aggregate);
DIOPI_API diopiError_t diopiAssignScoreWithkBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t points,
                                 diopiConstTensorHandle_t centers, diopiConstTensorHandle_t scores,
                                 diopiConstTensorHandle_t knn_idx, diopiTensorHandle_t grad_points,
                                 diopiTensorHandle_t grad_centers, diopiTensorHandle_t grad_scores,
                                 int64_t B, int64_t N0, int64_t N1, int64_t M, int64_t K, int64_t O,
                                 int64_t aggregate);

/**
 * \brief Find nearby points in spherical space.
 */
DIOPI_API diopiError_t diopiBallQuery(diopiContextHandle_t ctx, diopiConstTensorHandle_t new_xyz, diopiConstTensorHandle_t xyz, diopiTensorHandle_t idx, int64_t b, int64_t n, int64_t m,
                                      float min_radius, float max_radius, int64_t nsample);
DIOPI_API diopiError_t diopiStackBallQuery(diopiContextHandle_t ctx, diopiConstTensorHandle_t new_xyz, diopiConstTensorHandle_t new_xyz_batch_cnt, diopiConstTensorHandle_t xyz,
                                           diopiConstTensorHandle_t xyz_batch_cnt, diopiTensorHandle_t idx, float max_radius, int64_t nsample);

/**
 * \brief Calculate overlap between two set of bboxes.
 */
DIOPI_API diopiError_t diopiBboxOverlaps(diopiContextHandle_t ctx, diopiConstTensorHandle_t bboxes1, diopiConstTensorHandle_t bboxes2, diopiTensorHandle_t ious,
                        int64_t mode, bool aligned, int64_t offset);

/**
 * \brief Border align pooling layer.
 */
DIOPI_API diopiError_t diopiBorderAlign(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t boxes,
                               diopiTensorHandle_t output, diopiTensorHandle_t argmax_idx,
                               int64_t pool_size);
DIOPI_API diopiError_t diopiBorderAlignBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t boxes,
                                diopiConstTensorHandle_t argmax_idx, diopiTensorHandle_t grad_input,
                                int64_t pool_size);

/**
 * \brief Return intersection-over-union (Jaccard index) of boxes.
 */
DIOPI_API diopiError_t diopiBoxIouRotated(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes1, diopiConstTensorHandle_t boxes2, diopiTensorHandle_t ious,
                                          int64_t mode_flag, bool aligned);
DIOPI_API diopiError_t diopiBoxIouQuadri(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes1, diopiConstTensorHandle_t boxes2, diopiTensorHandle_t ious,
                                         int64_t mode_flag, bool aligned);

/**
 * \brief Content-Aware ReAssembly of FEatures
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
 * \brief This correlation operator works for optical flow correlation computation.
 */
DIOPI_API diopiError_t diopiCorrelation(diopiContextHandle_t ctx, diopiTensorHandle_t input1, diopiTensorHandle_t input2, diopiTensorHandle_t output, int64_t kH,
                         int64_t kW, int64_t patchH, int64_t patchW, int64_t padH, int64_t padW,
                         int64_t dilationH, int64_t dilationW, int64_t dilation_patchH,
                         int64_t dilation_patchW, int64_t dH, int64_t dW);
DIOPI_API diopiError_t diopiCorrelationBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiTensorHandle_t input1, diopiTensorHandle_t input2,
                          diopiTensorHandle_t grad_input1, diopiTensorHandle_t grad_input2, int64_t kH,
                          int64_t kW, int64_t patchH, int64_t patchW, int64_t padH, int64_t padW,
                          int64_t dilationH, int64_t dilationW, int64_t dilation_patchH,
                          int64_t dilation_patchW, int64_t dH, int64_t dW);

/**
 * \brief Deformable 2D convolution.
 */
DIOPI_API diopiError_t diopiDeformConv(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t weight, diopiTensorHandle_t offset,
                         diopiTensorHandle_t output, diopiTensorHandle_t columns, diopiTensorHandle_t ones, int64_t kW,
                         int64_t kH, int64_t dW, int64_t dH, int64_t padW, int64_t padH,
                         int64_t dilationW, int64_t dilationH, int64_t group,
                         int64_t deformable_group, int64_t im2col_step);
DIOPI_API diopiError_t diopiDeformConvBackwardInput(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t offset, diopiTensorHandle_t gradOutput,
                                diopiTensorHandle_t gradInput, diopiTensorHandle_t gradOffset,
                                diopiTensorHandle_t weight, diopiTensorHandle_t columns, int64_t kW, int64_t kH,
                                int64_t dW, int64_t dH, int64_t padW, int64_t padH,
                                int64_t dilationW, int64_t dilationH, int64_t group,
                                int64_t deformable_group, int64_t im2col_step);
DIOPI_API diopiError_t diopiDeformConvBackwardParameters(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t offset,
                                     diopiTensorHandle_t gradOutput, diopiTensorHandle_t gradWeight,
                                     diopiTensorHandle_t columns, diopiTensorHandle_t ones, int64_t kW,
                                     int64_t kH, int64_t dW, int64_t dH, int64_t padW, int64_t padH,
                                     int64_t dilationW, int64_t dilationH, int64_t group,
                                     int64_t deformable_group, float scale,
                                     int64_t im2col_step);

/**
 * \brief Deformable RoiPool
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
 * \brief SigmoidFocalLoss
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
 * \brief SoftmaxFocalLoss
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
 * \brief Uses iterative furthest point sampling to select a set of features whose corresponding points have the furthest distance.
 */
DIOPI_API diopiError_t diopiFurthestPointSampling(diopiContextHandle_t ctx, diopiTensorHandle_t points_tensor,
                                                diopiTensorHandle_t temp_tensor, diopiTensorHandle_t idx_tensor,
                                                int64_t b, int64_t n, int64_t m);
DIOPI_API diopiError_t diopiFurthestPointSamplingWithDist(diopiContextHandle_t ctx, diopiTensorHandle_t points_tensor,
                                                        diopiTensorHandle_t temp_tensor,
                                                        diopiTensorHandle_t idx_tensor, int64_t b,
                                                        int64_t n, int64_t m);

/**
 * \brief Calculate second order deviation.
 */
DIOPI_API diopiError_t diopiFusedBiasLeakyrelu(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input,
                                               diopiConstTensorHandle_t bias, diopiConstTensorHandle_t refer, int64_t act, int64_t grad, float alpha, float scale);

/**
 * \brief Gather points with given index.
 */
DIOPI_API diopiError_t diopiGatherPoints(diopiContextHandle_t ctx, diopiConstTensorHandle_t points, diopiConstTensorHandle_t idx, diopiTensorHandle_t out,
                                                  int64_t b, int64_t c, int64_t n, int64_t npoints);
DIOPI_API diopiError_t diopiGatherPointsBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t idx, diopiTensorHandle_t grad_points,
                                                   int64_t b, int64_t c, int64_t n, int64_t npoints);

/**
 * \brief Groups points with a ball query of radius.
 */
DIOPI_API diopiError_t diopiGroupPoints(diopiContextHandle_t ctx, diopiConstTensorHandle_t points, diopiConstTensorHandle_t idx, diopiTensorHandle_t out,
                                                 int64_t b, int64_t c, int64_t n, int64_t npoints, int64_t nsample);
DIOPI_API diopiError_t diopiGroupPointsBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t idx, diopiTensorHandle_t grad_points,
                                                  int64_t b, int64_t c, int64_t n, int64_t npoints, int64_t nsample);
DIOPI_API diopiError_t diopiStackGroupPoints(diopiContextHandle_t ctx,
                                     diopiConstTensorHandle_t features_tensor,
                                     diopiConstTensorHandle_t features_batch_cnt_tensor,
                                     diopiConstTensorHandle_t idx_tensor,
                                     diopiConstTensorHandle_t idx_batch_cnt_tensor,
                                     diopiTensorHandle_t out_tensor,
                                     int64_t b, int64_t c, int64_t m, int64_t nsample);
DIOPI_API diopiError_t diopiStackGroupPointsBackward(diopiContextHandle_t ctx,
                                      diopiConstTensorHandle_t grad_out_tensor,
                                      diopiConstTensorHandle_t idx_tensor,
                                      diopiConstTensorHandle_t idx_batch_cnt_tensor,
                                      diopiConstTensorHandle_t features_batch_cnt_tensor,
                                      diopiTensorHandle_t grad_features_tensor,
                                      int64_t b, int64_t c, int64_t m, int64_t n, int64_t nsample);

/**
 * \brief Calculate boxes BEV overlap.
 */
DIOPI_API diopiError_t diopiIou3dBoxesOverlapBev(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes_a, diopiConstTensorHandle_t boxes_b,
                                                            diopiTensorHandle_t ans_overlap,int64_t num_a, int64_t num_b);

/**
 * \brief 3D NMS function GPU implementation (for BEV boxes).
 */
DIOPI_API diopiError_t diopiIou3dNms3d(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes, diopiTensorHandle_t keep, diopiTensorHandle_t keep_num, float nms_overlap_thresh);

/**
 * \brief Normal 3D NMS function GPU implementation. The overlap of two boxes for
          IoU calculation is defined as the exact overlapping area of the two boxes
          WITH their yaw angle set to 0.
 */
DIOPI_API diopiError_t diopiIou3dNms3dNormal(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes, diopiTensorHandle_t keep, diopiTensorHandle_t keep_num, float nms_overlap_thresh);

/**
 * \brief KNN based on heap data structure.
 */
DIOPI_API diopiError_t diopiKnn(diopiContextHandle_t ctx, diopiTensorHandle_t xyz_tensor, diopiTensorHandle_t new_xyz_tensor, diopiTensorHandle_t idx_tensor,
                 diopiTensorHandle_t dist2_tensor, int64_t b, int64_t n, int64_t m, int64_t nsample);

/**
 * \brief A MaskedConv2d which inherits the official Conv2d.
 */
DIOPI_API diopiError_t diopiMaskedIm2col(diopiContextHandle_t ctx, diopiConstTensorHandle_t im, diopiConstTensorHandle_t mask_h_idx,
                                diopiConstTensorHandle_t mask_w_idx, diopiTensorHandle_t col,
                                int64_t kernel_h, int64_t kernel_w,
                                int64_t pad_h, int64_t pad_w);
DIOPI_API diopiError_t diopiMaskedCol2im(diopiContextHandle_t ctx, diopiConstTensorHandle_t col, diopiConstTensorHandle_t mask_h_idx,
                                diopiConstTensorHandle_t mask_w_idx, diopiTensorHandle_t im, int64_t height,
                                int64_t width, int64_t channels);

/**
 * \brief A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.
 */
DIOPI_API diopiError_t diopiModulatedDeformConv(
    diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t weight, diopiTensorHandle_t bias, diopiTensorHandle_t ones, diopiTensorHandle_t offset,
    diopiTensorHandle_t mask, diopiTensorHandle_t output, diopiTensorHandle_t columns, int64_t kernel_h, int64_t kernel_w,
    const int64_t stride_h, const int64_t stride_w, const int64_t pad_h, const int64_t pad_w,
    const int64_t dilation_h, const int64_t dilation_w, const int64_t group,
    const int64_t deformable_group, const bool with_bias);
DIOPI_API diopiError_t diopiModulatedDeformConvBackward(
    diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t weight, diopiTensorHandle_t bias, diopiTensorHandle_t ones, diopiTensorHandle_t offset,
    diopiTensorHandle_t mask, diopiTensorHandle_t columns, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
    diopiTensorHandle_t grad_bias, diopiTensorHandle_t grad_offset, diopiTensorHandle_t grad_mask, diopiTensorHandle_t grad_output,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w, int64_t pad_h,
    int64_t pad_w, int64_t dilation_h, int64_t dilation_w, int64_t group, int64_t deformable_group,
    const bool with_bias);

/**
 * \brief An attention module used in Deformable-Detr.
 */
DIOPI_API diopiError_t diopiMsDeformAttn(diopiContextHandle_t ctx, diopiTensorHandle_t* out,
                                   diopiConstTensorHandle_t value,
                                   diopiConstTensorHandle_t spatial_shapes,
                                   diopiConstTensorHandle_t level_start_index,
                                   diopiConstTensorHandle_t sampling_loc,
                                   diopiConstTensorHandle_t attn_weight,
                                   int64_t im2col_step);
DIOPI_API diopiError_t diopiMsDeformAttnBackward(
    diopiContextHandle_t ctx, diopiConstTensorHandle_t value, diopiConstTensorHandle_t spatial_shapes,
    diopiConstTensorHandle_t level_start_index, diopiConstTensorHandle_t sampling_loc,
    diopiConstTensorHandle_t attn_weight, diopiConstTensorHandle_t grad_output, diopiTensorHandle_t grad_value,
    diopiTensorHandle_t grad_sampling_loc, diopiTensorHandle_t grad_attn_weight, int64_t im2col_step);

/**
 * \brief NMS from mmcv. This function is modified from: https://github.com/pytorch/vision/
 */
DIOPI_API diopiError_t diopiNmsMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t dets,
                                    diopiConstTensorHandle_t scores, double iou_threshold, int64_t offset);

/**
 * \brief Performs non-maximum suppression (NMS) on the rotated boxes according to their intersection-over-union (IoU).
 */
DIOPI_API diopiError_t diopiNmsRotated(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t dets,
                                    diopiConstTensorHandle_t scores, diopiConstTensorHandle_t order_t, diopiConstTensorHandle_t dets_sorted,
                                    double iou_threshold, int64_t multi_label);

/**
 * \brief Find the box in which each point is
 */
DIOPI_API diopiError_t diopiPointsInBoxesPart(diopiContextHandle_t ctx, diopiConstTensorHandle_t boxes, diopiConstTensorHandle_t pts, diopiTensorHandle_t box_idx_of_points,
                                                    int64_t batch_size, int64_t boxes_num, int64_t pts_num);
DIOPI_API diopiError_t diopiPointsInBoxesAll(diopiContextHandle_t ctx, int64_t batch_size, int64_t boxes_num,
                                      int64_t pts_num, diopiConstTensorHandle_t boxes,
                                      diopiConstTensorHandle_t pts,
                                      diopiTensorHandle_t box_idx_of_points);

/**
 * \brief Psamask. Modified from https://github.com/hszhao/semseg/blob/master/lib/psa
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
 * \brief RoI align pooling layer.
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
 * \brief RoI align pooling layer for rotated proposals for mmcv.
 */
DIOPI_API diopiError_t diopiRoiAlignRotated(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t rois, diopiTensorHandle_t output,
                                    int64_t aligned_height, int64_t aligned_width,
                                    float spatial_scale, int64_t sampling_ratio,
                                    bool aligned, bool clockwise);
DIOPI_API diopiError_t diopiRoiAlignRotatedBackward(diopiContextHandle_t ctx, diopiTensorHandle_t top_grad, diopiTensorHandle_t rois,
                                     diopiTensorHandle_t bottom_grad, int64_t aligned_height,
                                     int64_t aligned_width, float spatial_scale,
                                     int64_t sampling_ratio, bool aligned,
                                     bool clockwise);

/**
 * \brief Rotation-invariant RoI align pooling layer for rotated proposals.
 */
DIOPI_API diopiError_t diopiRiroiAlignRotated(diopiContextHandle_t ctx, diopiTensorHandle_t features, diopiTensorHandle_t rois,
                                      diopiTensorHandle_t output, int64_t pooled_height,
                                      int64_t pooled_width, float spatial_scale,
                                      int64_t num_samples, int64_t num_orientations,
                                      bool clockwise);
DIOPI_API diopiError_t diopiRiroiAlignRotatedBackward(diopiContextHandle_t ctx, diopiTensorHandle_t top_grad, diopiTensorHandle_t rois,
                                       diopiTensorHandle_t bottom_grad, int64_t pooled_height,
                                       int64_t pooled_width, float spatial_scale,
                                       int64_t num_samples, int64_t num_orientations,
                                       bool clockwise);

/**
 * \brief Encode the geometry-specific features of each 3D proposal.
 */
DIOPI_API diopiError_t diopiRoiawarePool3d(diopiContextHandle_t ctx, int64_t boxes_num, int64_t pts_num, int64_t channels,
                                  int64_t max_pts_each_voxel, int64_t out_x, int64_t out_y,
                                  int64_t out_z, diopiConstTensorHandle_t rois,
                                  diopiConstTensorHandle_t pts, diopiConstTensorHandle_t pts_feature,
                                  diopiTensorHandle_t argmax, diopiTensorHandle_t pts_idx_of_voxels,
                                  diopiTensorHandle_t pooled_features, int64_t pool_method);

DIOPI_API diopiError_t diopiRoiawarePool3dBackward(diopiContextHandle_t ctx, int64_t boxes_num, int64_t out_x, int64_t out_y,
                                   int64_t out_z, int64_t channels,
                                   int64_t max_pts_each_voxel,
                                   diopiConstTensorHandle_t pts_idx_of_voxels,
                                   diopiConstTensorHandle_t argmax, diopiConstTensorHandle_t grad_out,
                                   diopiTensorHandle_t grad_in, int64_t pool_method);

/**
 * \brief Encode the geometry-specific features of each 3D proposal.
 */
DIOPI_API diopiError_t diopiRoipointPool3d(diopiContextHandle_t ctx, int64_t batch_size, int64_t pts_num, int64_t boxes_num,
                                  int64_t feature_in_len, int64_t sampled_pts_num,
                                  diopiConstTensorHandle_t xyz, diopiConstTensorHandle_t boxes3d,
                                  diopiConstTensorHandle_t pts_feature,
                                  diopiTensorHandle_t pooled_features,
                                  diopiTensorHandle_t pooled_empty_flag);

/**
 * \brief RoiPool.
 */
DIOPI_API diopiError_t diopiRoiPool(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t rois, diopiTensorHandle_t output,
                           diopiTensorHandle_t argmax, int64_t pooled_height, int64_t pooled_width,
                           float spatial_scale);
DIOPI_API diopiError_t diopiRoiPoolBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiTensorHandle_t rois, diopiTensorHandle_t argmax,
                            diopiTensorHandle_t grad_input, int64_t pooled_height,
                            int64_t pooled_width, float spatial_scale);

/**
 * \brief Scatters points into voxels, used in the voxel encoder with dynamic voxelization.
 */
DIOPI_API diopiError_t diopiDynamicPointToVoxel(diopiContextHandle_t ctx, diopiTensorHandle_t* outlist, diopiConstTensorHandle_t feats, diopiConstTensorHandle_t coors,
                                                int64_t reduce_type);
DIOPI_API diopiError_t diopiDynamicPointToVoxelBackward(
    diopiContextHandle_t ctx, diopiTensorHandle_t grad_feats, diopiConstTensorHandle_t grad_reduced_feats,
    diopiConstTensorHandle_t feats, diopiConstTensorHandle_t reduced_feats,
    diopiConstTensorHandle_t coors_idx, diopiConstTensorHandle_t reduce_count,
    int64_t reduce_type);

/**
 * \brief Synchronized Batch Normalization.
 */
DIOPI_API diopiError_t diopiSyncBnMean(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiTensorHandle_t mean);
DIOPI_API diopiError_t diopiSyncBnVar(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean,
                                      diopiTensorHandle_t var);
DIOPI_API diopiError_t diopiSyncBnOutput(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean,
                                 diopiConstTensorHandle_t var, diopiTensorHandle_t running_mean,
                                 diopiTensorHandle_t running_var, diopiConstTensorHandle_t weight,
                                 diopiConstTensorHandle_t bias, diopiTensorHandle_t norm, diopiTensorHandle_t std,
                                 diopiTensorHandle_t output, float eps, float momentum, int64_t group_size);
DIOPI_API diopiError_t diopiSyncBnParam(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t norm,
                                        diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias);
DIOPI_API diopiError_t diopiSyncBnBackwardData(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t weight,
                                diopiConstTensorHandle_t grad_weight, diopiConstTensorHandle_t grad_bias, diopiConstTensorHandle_t norm,
                                diopiConstTensorHandle_t std, diopiTensorHandle_t grad_input);

/**
 * \brief Performs weighted linear interpolation on 3 features.
 */
DIOPI_API diopiError_t diopiThreeInterpolate(diopiContextHandle_t ctx, diopiConstTensorHandle_t points, diopiConstTensorHandle_t idx,
                                             diopiConstTensorHandle_t weight, diopiTensorHandle_t out, int64_t b, int64_t c, int64_t m, int64_t n);
DIOPI_API diopiError_t diopiThreeInterpolateBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t idx,
                                     diopiConstTensorHandle_t weight, diopiTensorHandle_t grad_points, int64_t b, int64_t c, int64_t n, int64_t m);

/**
 * \brief Find the top-3 nearest neighbors of the target set from the source set.
 */
DIOPI_API diopiError_t diopiThreeNn(diopiContextHandle_t ctx, diopiConstTensorHandle_t unknown, diopiConstTensorHandle_t known, diopiTensorHandle_t dist2,
                                    diopiTensorHandle_t idx, int64_t b, int64_t n, int64_t m);

/**
 * \brief Temporal Interlace Shift.
 */
DIOPI_API diopiError_t diopiTinShift(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t shift, diopiTensorHandle_t output);
DIOPI_API diopiError_t diopiTinShiftBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiTensorHandle_t shift, diopiTensorHandle_t grad_input);

/**
 * \brief UpFRIDn for 2d features.
 */
DIOPI_API diopiError_t diopiUpfirdn2dOp(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input,
                           diopiConstTensorHandle_t kernel, int64_t up_x, int64_t up_y,
                           int64_t down_x, int64_t down_y, int64_t pad_x0, int64_t pad_x1,
                           int64_t pad_y0, int64_t pad_y1);

/**
 * \brief Convert kitti points(N, >=3) to voxels.
 */
DIOPI_API diopiError_t diopiHardVoxelize(diopiConstTensorHandle_t points,
                           diopiConstTensorHandle_t voxel_size,
                           diopiConstTensorHandle_t coors_range, diopiTensorHandle_t voxels,
                           diopiTensorHandle_t coors, diopiTensorHandle_t num_points_per_voxel,
                           diopiTensorHandle_t voxel_num, const int64_t max_points,
                           const int64_t max_voxels, const int64_t NDim,
                           const bool deterministic);
DIOPI_API diopiError_t diopiDynamicVoxelize(diopiConstTensorHandle_t points,
                              diopiConstTensorHandle_t voxel_size,
                              diopiConstTensorHandle_t coors_range, diopiTensorHandle_t coors,
                              const int64_t NDim);

/**
 * \brief Using the feature interpolation to obtain the position information
          correspond to the refined rotate anchors and reconstruct the feature maps
          in pixel-wise manner to achieve feature alignment.
 */
DIOPI_API diopiError_t diopiRotatedFeatureAlign(diopiContextHandle_t ctx, diopiConstTensorHandle_t features,
                                        diopiConstTensorHandle_t best_bboxes,
                                        float spatial_scale,
                                        int64_t points, diopiTensorHandle_t output);
DIOPI_API diopiError_t diopiRotatedFeatureAlignBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t top_grad,
                                         diopiConstTensorHandle_t best_bboxes,
                                         float spatial_scale,
                                         int64_t points, diopiTensorHandle_t bottom_grad);

/**
 * \brief Judging whether points are inside polygons, which is used in the ATSS assignment for the rotated boxes.
 */
DIOPI_API diopiError_t diopiPointsInPolygons(diopiContextHandle_t ctx, diopiConstTensorHandle_t points, diopiConstTensorHandle_t polygons,
                                     diopiTensorHandle_t output, int64_t rows,
                                     int64_t cols);

/**
 * \brief Sparse ops
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
 * \brief Find the smallest polygons that surrounds all points in the point sets.
 */
DIOPI_API diopiError_t diopiMinAreaPolygons(diopiContextHandle_t ctx, diopiConstTensorHandle_t pointsets, diopiTensorHandle_t polygons);

/**
 * \brief Encoding the orientation information and generating orientation-sensitive features.
 */
DIOPI_API diopiError_t diopiActiveRotatedFilter(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices, diopiTensorHandle_t output);
DIOPI_API diopiError_t diopiActiveRotatedFilterBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t indices, diopiTensorHandle_t grad_in);

/**
 * \brief Return generalized intersection-over-union (Jaccard index) between point sets and polygons.
 */
DIOPI_API diopiError_t diopiConvexIou(diopiContextHandle_t ctx, diopiConstTensorHandle_t pointsets, diopiConstTensorHandle_t polygons, diopiTensorHandle_t ious);
DIOPI_API diopiError_t diopiConvexGiou(diopiContextHandle_t ctx, diopiConstTensorHandle_t pointsets, diopiConstTensorHandle_t polygons, diopiTensorHandle_t output);

/**
 * \brief SortVertices.
 */
DIOPI_API diopiError_t diopiDiffIouRotatedSortVertices(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiTensorHandle_t vertices, diopiTensorHandle_t mask,
                                                   diopiTensorHandle_t num_valid);

/**
 * \brief This is an implementation of the 2D Chamfer Distance. It has been used in the paper `Oriented RepPoints for Aerial Object
 *        Detection (CVPR 2022) <https://arxiv.org/abs/2105.11111>_`.
 */
DIOPI_API diopiError_t diopiChamferDistance(diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz1_in, diopiConstTensorHandle_t xyz2_in, diopiTensorHandle_t dist1_out,
                                            diopiTensorHandle_t dist2_out, diopiTensorHandle_t idx1_out, diopiTensorHandle_t idx2_out);
DIOPI_API diopiError_t diopiChamferDistanceBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz1, diopiConstTensorHandle_t xyz2,
                                            diopiConstTensorHandle_t idx1, diopiConstTensorHandle_t idx2, diopiConstTensorHandle_t grad_dist1, diopiConstTensorHandle_t grad_dist2,
                                            diopiTensorHandle_t grad_xyz1, diopiTensorHandle_t grad_xyz2);

/**
 * \brief The operation of precision RoI pooling. The implementation of PrRoIPool is modified from https://github.com/vacancy/PreciseRoIPooling/
 */
DIOPI_API diopiError_t diopiPrroiPool(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t rois, diopiTensorHandle_t output,
                                      int64_t pooled_height, int64_t pooled_width, float spatial_scale);
DIOPI_API diopiError_t diopiPrroiPoolbackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiTensorHandle_t rois, diopiTensorHandle_t grad_input,
                                              int64_t pooled_height, int64_t pooled_width, float spatial_scale);
DIOPI_API diopiError_t diopiPrroiPoolCoorBackward(diopiContextHandle_t ctx, diopiTensorHandle_t output, diopiTensorHandle_t grad_output, diopiTensorHandle_t input, diopiTensorHandle_t rois, diopiTensorHandle_t grad_rois,
                                                  int64_t pooled_height, int64_t pooled_width, float spatial_scale);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_MMCV_H_