/**
 * @file
 * @brief DIOPI 数据类型头文件
 * @author sensetime
 * @version 1.0.0 
 * @copyright  (c) 2022, SenseTime Inc.
 */

#ifndef _PROJECT_DIOPERATOR_INTERFACE_RT_H_
#define _PROJECT_DIOPERATOR_INTERFACE_RT_H_

#include <stdint.h>

#ifndef DIOPI_ATTR_WEAK
#define DIOPI_API
#else
#define DIOPI_API __attribute__((weak))
#endif
#define DIOPI_RT_API

#if defined(__cplusplus)
extern "C" {
#endif

#define DIOPI_VER_MAJOR 1
#define DIOPI_VER_MINOR 0
#define DIOPI_VER_PATCH 0
#define DIOPI_VERSION   (DIOPI_VER_MAJOR * 1000 + DIOPI_VER_MINOR * 100 + DIOPI_VER_PATCH)

/**
 * @brief 数组结构体
 * @brief 用于存储多元数据，stride、padding以及dilation等的数据类型
 */
typedef struct diopiSize_t_ {
    /// @brief 数据指针
    const int64_t* data;
    /// @brief 数据长度
    int64_t  len;

#if defined(__cplusplus)
    diopiSize_t_() : data(nullptr), len(0) {}
    diopiSize_t_(const int64_t* d, int64_t l) : data(d), len(l) {}
#endif  // __cplusplus
} diopiSize_t;

/**
 * @brief diopi函数执行结果类型枚举
 * 共有13种枚举类型
 */
typedef enum {
    /// @brief 执行成功
    diopiSuccess                                      = 0,
    /// @brief 出错
    diopiErrorOccurred                                = 1,
    /// @brief 未初始化
    diopiNotInited                                    = 2,
    /// @brief 未注册的流创建函数
    diopiNoRegisteredStreamCreateFunction             = 3,
    /// @brief 未注册的流销毁函数
    diopiNoRegisteredStreamDestoryFunction            = 4,
    /// @brief 未注册的流同步函数
    diopiNoRegisteredStreamSyncFunction               = 5,
    /// @brief 未注册的设备内存分配函数
    diopiNoRegisteredDeviceMemoryMallocFunction       = 6,
    /// @brief 未注册的设备内存释放函数
    diopiNoRegisteredDeviceMemoryFreeFunction         = 7,
    /// @brief 未注册的设备之间内存拷贝函数
    diopiNoRegisteredDevice2DdeviceMemoryCopyFunction = 8,
    /// @brief 未注册的设备到主机的内存拷贝函数
    diopiNoRegisteredDevice2HostMemoryCopyFunction    = 9,
    /// @brief 未注册的主机到设备的内存拷贝函数
    diopiNoRegisteredHost2DeviceMemoryCopyFunction    = 10,
    /// @brief 未注册的最终错误获取函数
    diopiNoRegisteredGetLastErrorFunction             = 11,
    /// @brief 不支持5维
    diopi5DNotSupported                               = 12,
    /// @brief 不支持的数据类型
    diopiDtypeNotSupported                            = 1000,
} diopiError_t;

/**
 * @brief 主机设备类型枚举
 * 共有2种枚举类型
 */
typedef enum {
    /// @brief 主机类型
    diopi_host   = 0,
    /// @brief 设备类型
    diopi_device = 1,
} diopiDevice_t;

/**
 * @brief 数据类型枚举
 * 共有13种枚举类型
 */
typedef enum {
    /// @brief 8位整型
    diopi_dtype_int8     = 0,
    /// @brief 无符号8位整型
    diopi_dtype_uint8    = 1,
    /// @brief 16位整型
    diopi_dtype_int16    = 2,
    /// @brief 无符号16位整型
    diopi_dtype_uint16   = 3,
    /// @brief 32位整型
    diopi_dtype_int32    = 4,
    /// @brief 无符号32位整型
    diopi_dtype_uint32   = 5,
    /// @brief 64位整型
    diopi_dtype_int64    = 6,
    /// @brief 无符号64位整型
    diopi_dtype_uint64   = 7,
    /// @brief 16位浮点型
    diopi_dtype_float16  = 8,
    /// @brief 32位浮点型
    diopi_dtype_float32  = 9,
    /// @brief 64位浮点型
    diopi_dtype_float64  = 10,
    /// @brief 布尔型
    diopi_dtype_bool     = 11,
    /// @brief 16位浮点型，指数位与float32一样多，为8位，小数位比float16少，为7位
    diopi_dtype_bfloat16 = 12,
    /// @brief 32位浮点型，英伟达提出的代替float32的单精度浮点格式，包含8位指数位，10位小数位
    diopi_dtype_tfloat32 = 13,
} diopiDtype_t;

/**
 * @brief 上下文环境结构体
 * 用于表示上下文环境
 */
struct diopiContext;

/**
 * @brief 上下文环境指针
 * 用于指向上下文环境
 */
typedef struct diopiContext* diopiContextHandle_t;

/**
 * @brief 张量结构体
 * 用于表示张量对象
 */
struct diopiTensor;

/**
 * @brief 张量指针
 * 用于指向张量
 */
typedef struct diopiTensor* diopiTensorHandle_t;

/**
 * @brief 张量常量指针
 * 用于指向常量张量
 */
typedef const struct diopiTensor* diopiConstTensorHandle_t;

/**
 * @brief 流指针
 * 用于指向流
 */
typedef void* diopiStreamHandle_t;

/**
 * @brief 获取版本
 */
extern DIOPI_API const char* diopiGetVersion();

/**
 * @brief 获取张量数据
 */
extern DIOPI_RT_API diopiError_t diopiGetTensorData(diopiTensorHandle_t* th, void**);

/**
 * @brief 获取张量常量数据
 */
extern DIOPI_RT_API diopiError_t diopiGetTensorDataConst(diopiConstTensorHandle_t* th, const void**);

/**
 * @brief 获取张量形状
 */
extern DIOPI_RT_API diopiError_t diopiGetTensorShape(diopiConstTensorHandle_t th, diopiSize_t* size);

/**
 * @brief 获取张量的步长
 */
extern DIOPI_RT_API diopiError_t diopiGetTensorStride(diopiConstTensorHandle_t th, diopiSize_t* stride);

/**
 * @brief 获取张量的数据类型
 */
extern DIOPI_RT_API diopiError_t diopiGetTensorDtype(diopiConstTensorHandle_t th, diopiDtype_t* dtype);

/**
 * @brief 获取张量所在设备类型
 */
extern DIOPI_RT_API diopiError_t diopiGetTensorDevice(diopiConstTensorHandle_t th, diopiDevice_t* device);

/**
 * @brief 获取张量数据量大小
 */
extern DIOPI_RT_API diopiError_t diopiGetTensorNumel(diopiConstTensorHandle_t th, int64_t* numel);

/**
 * @brief 获取张量单元素所占存储空间大小
 */
extern DIOPI_RT_API diopiError_t diopiGetTensorElemSize(diopiConstTensorHandle_t th, int64_t* itemsize);

/**
 * @brief 获取处理流
 */
extern DIOPI_RT_API diopiError_t diopiGetStream(diopiContextHandle_t ctx, diopiStreamHandle_t* stream);

/**
 * @brief 创建张量
 */
extern DIOPI_RT_API diopiError_t diopiRequireTensor(diopiContextHandle_t ctx, diopiTensorHandle_t* tensor,
                                                 const diopiSize_t* size, const diopiSize_t* stride,
                                                 const diopiDtype_t dtype, const diopiDevice_t device);

/**
 * @brief 创建张量
 */
extern DIOPI_RT_API diopiError_t diopiRequireBuffer(diopiContextHandle_t ctx, diopiTensorHandle_t* tensor,
                                                 int64_t num_bytes, diopiDevice_t device);


#if defined(__cplusplus)
}
#endif

#endif   // _PROJECT_DIOPERATOR_INTERFACE_RT_H_
