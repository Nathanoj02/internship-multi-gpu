#ifndef DEFINITIONS_HPP
#define DEFINITIONS_HPP

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define BLOCK_SIZE 32

#define BM 64
#define BN 64
#define BK 8
#define TM 8

#define BN2D 64
#define BK2D 8
#define BM2D 64
#define TM2D 4
#define TN2D 4

#define BNWARP 64
#define BKWARP 8
#define BMWARP 64
#define TMWARPS 2
#define TNWARPS 2

#define WARPNUM 16
#define WARPSN 2
#define WARPSM 8
#define WN 32
#define WM 8

#define WARPSIZE 32
#define WSUBN 16
#define WSUBM 2
#define WNITER 1
#define WMITER 2

#endif // DEFINITIONS_HPP