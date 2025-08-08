#pragma once

#include <stdint.h>
#include <string.h>

// SM3常量
#define Tj15 0x79cc4519
#define Tj63 0x7a879d8a

#define ROTL(x,n) (((x) << (n)) | ((x) >> (32 - (n))))

#define FFj(x,y,z,j) ((j<16)?((x)^(y)^(z)):(((x)&(y))|((x)&(z))|((y)&(z))))
#define GGj(x,y,z,j) ((j<16)?((x)^(y)^(z)):(((x)&(y))|((~(x))&(z))))

#define P0(x) ((x) ^ ROTL((x), 9) ^ ROTL((x), 17))
#define P1(x) ((x) ^ ROTL((x), 15) ^ ROTL((x), 23))

// 初始值
static const uint32_t IV[8] = {
    0x7380166f,
    0x4914b2b9,
    0x172442d7,
    0xda8a0600,
    0xa96f30bc,
    0x163138aa,
    0xe38dee4d,
    0xb0fb0e4e
};

// 循环展开优化后的轮函数宏定义
#define ROUND(A,B,C,D,E,F,G,H,Wj,WjP,Tj,j) do { \
    uint32_t SS1 = ROTL((ROTL(A,12) + E + ROTL(Tj, j)) & 0xFFFFFFFF, 7); \
    uint32_t SS2 = SS1 ^ ROTL(A, 12); \
    uint32_t TT1 = (FFj(A,B,C,j) + D + SS2 + WjP) & 0xFFFFFFFF; \
    uint32_t TT2 = (GGj(E,F,G,j) + H + SS1 + Wj) & 0xFFFFFFFF; \
    D = C; C = ROTL(B, 9); B = A; A = TT1; \
    H = G; G = ROTL(F, 19); F = E; E = P0(TT2); \
} while(0)

void sm3_hash(const uint8_t * data, size_t len, uint8_t hash[32]);
void sm3_compress(uint32_t V[8], const uint8_t block[64]);
void sm3_length_extension_attack(const uint8_t old_hash[32], size_t original_len, const uint8_t * append, size_t append_len, uint8_t new_hash[32]);
size_t sm3_padding(const uint8_t * msg, size_t len, uint8_t * *padded);

typedef struct MerkleTree {
    uint8_t** leaves;
    size_t leaf_count;
    uint8_t* root;
} MerkleTree;

MerkleTree* build_merkle_tree(uint8_t** leaf_hashes, size_t count);
size_t get_merkle_proof(uint8_t** leaves, size_t count, size_t index, uint8_t*** proof_out);
int verify_merkle_proof(const uint8_t* leaf, size_t index, size_t total, uint8_t** proof, size_t depth, const uint8_t* expected_root);
