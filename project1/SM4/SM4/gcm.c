#include <stdint.h>
#include <string.h>
#include "sm4.h"

#define BLOCK_SIZE 16
#define GCM_UNIT 128

static void gf_mul(const uint8_t X[16], const uint8_t Y[16], uint8_t result[16]) {
    uint8_t Z[16] = { 0 };
    uint8_t V[16];
    memcpy(V, Y, 16);
    for (int i = 0; i < 128; ++i) {
        if ((X[i / 8] >> (7 - (i % 8))) & 1) {
            for (int j = 0; j < 16; ++j)
                Z[j] ^= V[j];
        }
        int carry = V[15] & 1;
        for (int j = 15; j > 0; --j)
            V[j] = (V[j] >> 1) | ((V[j - 1] & 1) << 7);
        V[0] >>= 1;
        if (carry)
            V[0] ^= 0xe1;
    }
    memcpy(result, Z, 16);
}

static void xor_block(uint8_t* out, const uint8_t* a, const uint8_t* b) {
    for (int i = 0; i < 16; ++i) out[i] = a[i] ^ b[i];
}

static void increment_counter_8(uint8_t ctr[8][16]) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 15; j >= 12; --j) {
            if (++ctr[i][j]) break;
        }
    }
}

static void ghash(const uint8_t* H, const uint8_t* aad, size_t aad_len,
    const uint8_t* ciphertext, size_t ct_len, uint8_t tag[16]) {
    uint8_t Y[16] = { 0 };
    uint8_t block[16];
    for (size_t i = 0; i < aad_len; i += 16) {
        memset(block, 0, 16);
        size_t len = (aad_len - i >= 16) ? 16 : aad_len - i;
        memcpy(block, aad + i, len);
        xor_block(Y, Y, block);
        gf_mul(Y, H, Y);
    }
    for (size_t i = 0; i < ct_len; i += 16) {
        memset(block, 0, 16);
        size_t len = (ct_len - i >= 16) ? 16 : ct_len - i;
        memcpy(block, ciphertext + i, len);
        xor_block(Y, Y, block);
        gf_mul(Y, H, Y);
    }
    uint8_t len_block[16] = { 0 };
    uint64_t aad_bits = aad_len * 8;
    uint64_t ct_bits = ct_len * 8;
    for (int i = 0; i < 8; ++i) {
        len_block[7 - i] = (aad_bits >> (i * 8)) & 0xff;
        len_block[15 - i] = (ct_bits >> (i * 8)) & 0xff;
    }
    xor_block(Y, Y, len_block);
    gf_mul(Y, H, Y);
    memcpy(tag, Y, 16);
}

void gcm_encrypt(const uint32_t* key, const uint8_t iv[12], const uint8_t* aad, size_t aad_len, const uint8_t plaintext[128], uint8_t ciphertext[128], uint8_t tag[16]) {
    uint32_t round_key[32];
    key_expansion(key, round_key);

    uint8_t H[16] = { 0 };
    uint8_t zero[128] = { 0 };
    encrypt(zero, zero, round_key);
    memcpy(H, zero, 16);

    uint8_t ctr[8][16] = { {0} };
    for (int i = 0; i < 8; ++i) {
        memcpy(ctr[i], iv, 12);
        ctr[i][15] = 1 + i;
    }

    uint8_t stream[128];
    encrypt((uint8_t*)ctr, stream, round_key);

    for (int i = 0; i < 128; ++i) {
        ciphertext[i] = plaintext[i] ^ stream[i];
    }

    uint8_t S[16];
    ghash(H, aad, aad_len, ciphertext, 128, S);

    uint8_t Y0[16] = { 0 };
    memcpy(Y0, iv, 12);
    Y0[15] = 0x01;

    uint8_t E_Y0[128] = { 0 };
    encrypt(Y0, E_Y0, round_key);
    xor_block(tag, S, E_Y0);
}

int gcm_decrypt(const uint32_t* key, const uint8_t iv[12], const uint8_t* aad, size_t aad_len, const uint8_t ciphertext[128], const uint8_t tag[16], uint8_t plaintext[128]) {
    uint32_t round_key[32];
    key_expansion((const uint32_t*)key, round_key);

    uint8_t H[16] = { 0 };
    uint8_t zero[128] = { 0 };
    encrypt(zero, zero, round_key);
    memcpy(H, zero, 16);

    uint8_t ctr[8][16] = { {0} };
    for (int i = 0; i < 8; ++i) {
        memcpy(ctr[i], iv, 12);
        ctr[i][15] = 1 + i;
    }

    uint8_t stream[128];
    encrypt((uint8_t*)ctr, stream, round_key);

    for (int i = 0; i < 128; ++i) {
        plaintext[i] = ciphertext[i] ^ stream[i];
    }

    uint8_t S[16];
    ghash(H, aad, aad_len, ciphertext, 128, S);

    uint8_t Y0[16] = { 0 };
    memcpy(Y0, iv, 12);
    Y0[15] = 0x01;

    uint8_t E_Y0[128] = { 0 };
    encrypt(Y0, E_Y0, round_key);

    uint8_t expected_tag[16];
    xor_block(expected_tag, S, E_Y0);

    return memcmp(expected_tag, tag, 16) == 0;
}