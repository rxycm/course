#include"sm4.h"

void key_expansion(const uint32_t* main_key, uint32_t* round_key) {
    static const uint32_t FK[4] = { 0xa3b1bac6, 0x56aa3350, 0x677d9197, 0xb27022dc };

    static const uint32_t CK[32] = {
            0x00070E15, 0x1C232A31, 0x383F464D, 0x545B6269, 0x70777E85, 0x8C939AA1, 0xA8AFB6BD, 0xC4CBD2D9,
            0xE0E7EEF5, 0xFC030A11, 0x181F262D, 0x343B4249, 0x50575E65, 0x6C737A81, 0x888F969D, 0xA4ABB2B9,
            0xC0C7CED5, 0xDCE3EAF1, 0xF8FF060D, 0x141B2229, 0x30373E45, 0x4C535A61, 0x686F767D, 0x848B9299,
            0xA0A7AEB5, 0xBCC3CAD1, 0xD8DFE6ED, 0xF4FB0209, 0x10171E25, 0x2C333A41, 0x484F565D, 0x646B7279
    };

    word x[4] = { 0 };
    word X;
    //初始化
    for (uint8_t i = 0; i < 4; i++) {
        x[i].w = main_key[i] ^ FK[i];
    }
    //32轮变换
    for (int i = 0; i < 32; i++) {

        X.w = x[(i + 1) & 3].w ^ x[(i + 2) & 3].w ^ x[(i + 3) & 3].w ^ CK[i];
        word t;
        //s盒
        t.b[0] = sm4_sbox[X.b[0]];
        t.b[1] = sm4_sbox[X.b[1]];
        t.b[2] = sm4_sbox[X.b[2]];
        t.b[3] = sm4_sbox[X.b[3]];
        //线性变换
        t.w = t.w ^ ROT32L(t.w, 13) ^ ROT32L(t.w, 23);
        //移位
        int j = i & 3;
        x[j].w = x[j].w ^ t.w;
        round_key[i] = x[j].w;
    }
}


static uint32_t T_transform(word X) {
    word t;
    t.b[0] = sm4_sbox[X.b[0]];
    t.b[1] = sm4_sbox[X.b[1]];
    t.b[2] = sm4_sbox[X.b[2]];
    t.b[3] = sm4_sbox[X.b[3]];
    t.w ^= ROT32L(t.w, 2) ^ ROT32L(t.w, 10) ^ ROT32L(t.w, 18) ^ ROT32L(t.w, 24);
    return t.w;
}
/*
void encrypt(const uint32_t* plain, uint32_t* cipher, const uint32_t* round_key) {
    uint32_t X[4];
    for (uint8_t i = 0; i < 4; i++) {
        X[i] = plain[i];
    }

    for (int i = 0; i < 32; i++) {
        word tmp;
        tmp.w = X[(i + 1) & 3] ^ X[(i + 2) & 3] ^ X[(i + 3) & 3] ^ round_key[i];
        X[i & 3] ^= T_transform(tmp);
    }

    for (uint8_t i = 0; i < 4; i++) {
        cipher[i] = X[3 - i];
    }
}
*/


void encrypt(uint8_t* m, uint8_t* c, uint32_t* round_key) {
    __m256i X[4], tmp[4], mask;
    mask = _mm256_set1_epi32(0xFF);
    //加载数据
    tmp[0] = _mm256_loadu_si256((const __m256i*)m);
    tmp[1] = _mm256_loadu_si256((const __m256i*)m + 1);
    tmp[2] = _mm256_loadu_si256((const __m256i*)m + 2);
    tmp[3] = _mm256_loadu_si256((const __m256i*)m + 3);
    //合并
    X[0] = PACK0(tmp[0], tmp[1], tmp[2], tmp[3]);
    X[1] = PACK1(tmp[0], tmp[1], tmp[2], tmp[3]);
    X[2] = PACK2(tmp[0], tmp[1], tmp[2], tmp[3]);
    X[3] = PACK3(tmp[0], tmp[1], tmp[2], tmp[3]);

    __m256i index =_mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    X[0] = _mm256_shuffle_epi8(X[0], index);
    X[1] = _mm256_shuffle_epi8(X[1], index);
    X[2] = _mm256_shuffle_epi8(X[2], index);
    X[3] = _mm256_shuffle_epi8(X[3], index);
    // 32轮迭代
    for (int i = 0; i < 32; i++) {
        __m256i k =_mm256_set1_epi32(round_key[i]);

        //查表
        tmp[0] = _mm256_xor_si256(_mm256_xor_si256(X[1], X[2]),_mm256_xor_si256(X[3], k));
        tmp[1] = _mm256_xor_si256(X[0], _mm256_i32gather_epi32((const int*)TABLE0,_mm256_and_si256(tmp[0], mask), 4));
        tmp[0] = _mm256_srli_epi32(tmp[0], 8);
        tmp[1] = _mm256_xor_si256(tmp[1], _mm256_i32gather_epi32((const int*)TABLE1, _mm256_and_si256(tmp[0], mask), 4));
        tmp[0] = _mm256_srli_epi32(tmp[0], 8);
        tmp[1] = _mm256_xor_si256(tmp[1], _mm256_i32gather_epi32((const int*)TABLE2, _mm256_and_si256(tmp[0], mask), 4));
        tmp[0] = _mm256_srli_epi32(tmp[0], 8);
        tmp[1] = _mm256_xor_si256(tmp[1], _mm256_i32gather_epi32((const int*)TABLE3, _mm256_and_si256(tmp[0], mask), 4));

        X[0] = X[1];
        X[1] = X[2];
        X[2] = X[3];
        X[3] = tmp[1];
    }

    X[0] = _mm256_shuffle_epi8(X[0], index);
    X[1] = _mm256_shuffle_epi8(X[1], index);
    X[2] = _mm256_shuffle_epi8(X[2], index);
    X[3] = _mm256_shuffle_epi8(X[3], index);
    //恢复分组
    _mm256_storeu_si256((__m256i*)c + 0, PACK0(X[3], X[2], X[1], X[0]));
    _mm256_storeu_si256((__m256i*)c + 1, PACK1(X[3], X[2], X[1], X[0]));
    _mm256_storeu_si256((__m256i*)c + 2, PACK2(X[3], X[2], X[1], X[0]));
    _mm256_storeu_si256((__m256i*)c + 3, PACK3(X[3], X[2], X[1], X[0]));
}