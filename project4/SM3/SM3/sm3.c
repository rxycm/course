#include"sm3.h"

void sm3_compress(uint32_t V[8], const uint8_t block[64]) {
    uint32_t W[68], Wp[64];
    uint32_t A, B, C, D, E, F, G, H;

    // 消息分组
    for (int i = 0; i < 16; ++i) {
        W[i] = ((uint32_t)block[i * 4 + 0] << 24) |
            ((uint32_t)block[i * 4 + 1] << 16) |
            ((uint32_t)block[i * 4 + 2] << 8) |
            ((uint32_t)block[i * 4 + 3]);
    }

    // 消息扩展
    for (int j = 16; j < 68; ++j) {
        W[j] = P1(W[j - 16] ^ W[j - 9] ^ ROTL(W[j - 3], 15))
            ^ ROTL(W[j - 13], 7) ^ W[j - 6];
    }
    for (int j = 0; j < 64; ++j) {
        Wp[j] = W[j] ^ W[j + 4];
    }

    // 压缩过程初始化
    A = V[0]; B = V[1]; C = V[2]; D = V[3];
    E = V[4]; F = V[5]; G = V[6]; H = V[7];

    // 循环展开压缩函数
    for (int j = 0; j < 64; ++j) {
        uint32_t Tj = (j < 16) ? Tj15 : Tj63;
        ROUND(A, B, C, D, E, F, G, H, W[j], Wp[j], Tj, j);
    }

    // 结果反馈
    V[0] ^= A; V[1] ^= B; V[2] ^= C; V[3] ^= D;
    V[4] ^= E; V[5] ^= F; V[6] ^= G; V[7] ^= H;
}

void sm3_hash(const uint8_t* data, size_t len, uint8_t hash[32]) {
    uint64_t total_len = len * 8;
    uint8_t block[64];
    uint32_t V[8];
    memcpy(V, IV, sizeof(IV));

    // 处理完整块
    while (len >= 64) {
        sm3_compress(V, data);
        data += 64;
        len -= 64;
    }

    // 填充最后的块
    memset(block, 0, 64);
    memcpy(block, data, len);
    block[len] = 0x80;

    if (len >= 56) {
        sm3_compress(V, block);
        memset(block, 0, 64);
    }

    for (int i = 0; i < 8; ++i)
        block[56 + i] = (total_len >> ((7 - i) * 8)) & 0xFF;

    sm3_compress(V, block);

    // 输出hash值
    for (int i = 0; i < 8; ++i) {
        hash[i * 4 + 0] = (V[i] >> 24) & 0xFF;
        hash[i * 4 + 1] = (V[i] >> 16) & 0xFF;
        hash[i * 4 + 2] = (V[i] >> 8) & 0xFF;
        hash[i * 4 + 3] = V[i] & 0xFF;
    }
}