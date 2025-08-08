#include"sm3.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>


// 将hash（32字节）还原为8个初始状态V
void hash_to_IV(const uint8_t hash[32], uint32_t V[8]) {
    for (int i = 0; i < 8; ++i) {
        V[i] = ((uint32_t)hash[i * 4] << 24) |
            ((uint32_t)hash[i * 4 + 1] << 16) |
            ((uint32_t)hash[i * 4 + 2] << 8) |
            ((uint32_t)hash[i * 4 + 3]);
    }
}

// 构造 SM3 的消息填充
size_t sm3_padding(const uint8_t* msg, size_t len, uint8_t** pad) {
    size_t total = ((len + 9 + 63) / 64) * 64;
    *pad = (uint8_t*)calloc(1, total);
    memcpy(*pad, msg, len);
    (*pad)[len] = 0x80;
    uint64_t bitlen = len * 8;
    for (int i = 0; i < 8; ++i)
        (*pad)[total - 8 + i] = (bitlen >> ((7 - i) * 8)) & 0xFF;
    return total;
}

// 执行扩展攻击：以已知哈希为初始值，追加恶意消息
void sm3_length_extension_attack(const uint8_t old_hash[32], size_t original_len, const uint8_t* append, size_t append_len, uint8_t new_hash[32]) {
    uint32_t V[8];
    hash_to_IV(old_hash, V);

    size_t forged_offset = ((original_len + 9 + 63) / 64) * 64;
    size_t total_len = forged_offset + append_len;

    size_t new_padded_len = ((total_len + 9 + 63) / 64) * 64;
    uint8_t* buffer = (uint8_t*)calloc(1, new_padded_len);

    // 将追加数据放在"伪造消息末尾"
    memcpy(buffer + forged_offset, append, append_len);

    // 构造追加数据的 padding
    buffer[forged_offset + append_len] = 0x80;
    uint64_t bitlen = (total_len) * 8;
    for (int i = 0; i < 8; ++i)
        buffer[new_padded_len - 8 + i] = (bitlen >> ((7 - i) * 8)) & 0xff;

    // 从 forged_offset 开始输入压缩函数
    for (size_t i = forged_offset; i < new_padded_len; i += 64)
        sm3_compress(V, buffer + i);

    for (int i = 0; i < 8; ++i) {
        new_hash[i * 4 + 0] = (V[i] >> 24) & 0xff;
        new_hash[i * 4 + 1] = (V[i] >> 16) & 0xff;
        new_hash[i * 4 + 2] = (V[i] >> 8) & 0xff;
        new_hash[i * 4 + 3] = V[i] & 0xff;
    }

    free(buffer);
}
