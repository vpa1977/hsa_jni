/* HEADER FILE GENERATED BY cloc VERSION 0.7.4 */
/* THIS FILE:  /home/bsp/hsa_jni/kernels/nominal_distance.h  */
/* INPUT FILE: /home/bsp/hsa_jni/kernels/nominal_distance.cl  */
#ifdef __cplusplus
#define _CPPSTRING_ "C" 
#endif
#ifndef __cplusplus
#define _CPPSTRING_ 
#endif
#ifndef __SNACK_DEFS
typedef struct transfer_t Transfer_t;
struct transfer_t { int nargs ; size_t* rsrvd1; size_t* rsrvd2 ; size_t* rsrvd3; } ;
typedef struct lparm_t Launch_params_t;
struct lparm_t { int ndim; size_t gdims[3]; size_t ldims[3]; Transfer_t transfer  ;} ;
#define __SNACK_DEFS
#endif
extern _CPPSTRING_ void nominal_distance(const double* temp,const int len,double* result, const Launch_params_t lparm);