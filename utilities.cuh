//#include "timer.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include  "device_atomic_functions.h"
#include "helper_functions.h"
#include "helper_cuda.h"
#define uchar unsigned char
#define uint unsigned int
#define uf  unsigned float
#define Windowx 32
#define Windowy 32
#define m_nImage 128
#define Pi 3.1415926535897f
#define UMUL(a, b) ( (a) * (b) )
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )
 uint iDivUp(uint a, uint b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}