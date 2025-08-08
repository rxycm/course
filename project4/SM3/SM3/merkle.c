#include"sm3.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>




// 哈希拼接: H = Hash(left || right)
void hash_concat(const uint8_t* left, const uint8_t* right, uint8_t hash[32]) {
    uint8_t buf[64];
    memcpy(buf, left, 32);
    memcpy(buf + 32, right, 32);
    sm3_hash(buf, 64, hash);
}

// 构建 Merkle 树，返回根节点
uint8_t* build_merkle_root(uint8_t** leaves, size_t count) {
    if (count == 0) return NULL;
    if (count == 1) {
        uint8_t* root = (uint8_t*)malloc(32);
        memcpy(root, leaves[0], 32);
        return root;
    }

    size_t next_level = (count + 1) / 2;
    uint8_t** parents = (uint8_t**)malloc(next_level * sizeof(uint8_t*));
    for (size_t i = 0; i < next_level; ++i) {
        uint8_t* hash = (uint8_t*)malloc(32);
        if (2 * i + 1 < count)
            hash_concat(leaves[2 * i], leaves[2 * i + 1], hash);
        else
            hash_concat(leaves[2 * i], leaves[2 * i], hash); // duplicate last
        parents[i] = hash;
    }

    uint8_t* root = build_merkle_root(parents, next_level);
    for (size_t i = 0; i < next_level; ++i) free(parents[i]);
    free(parents);
    return root;
}

// 构建 Merkle 树对象
MerkleTree* build_merkle_tree(uint8_t** leaf_hashes, size_t count) {
    MerkleTree* tree = (MerkleTree*)malloc(sizeof(MerkleTree));
    tree->leaf_count = count;
    tree->leaves = leaf_hashes;
    tree->root = build_merkle_root(leaf_hashes, count);
    return tree;
}

// 获取某个叶子的存在性证明（Merkle路径）
size_t get_merkle_proof(uint8_t** leaves, size_t count, size_t index, uint8_t*** proof_out) {
    if (index >= count) return 0;

    size_t depth = 0;
    size_t level_size = count;
    size_t pos = index;
    uint8_t** proof = (uint8_t**)malloc(32 * sizeof(uint8_t*)); // max depth = 32
    while (level_size > 1) {
        size_t sibling = (pos % 2 == 0) ? pos + 1 : pos - 1;
        if (sibling >= level_size) sibling = pos; // duplicate if odd

        uint8_t* sibling_hash = (uint8_t*)malloc(32);
        memcpy(sibling_hash, leaves[sibling], 32);
        proof[depth++] = sibling_hash;

        // prepare next level
        size_t next_level = (level_size + 1) / 2;
        uint8_t** parents = (uint8_t**)malloc(next_level * sizeof(uint8_t*));
        for (size_t i = 0; i < next_level; ++i) {
            parents[i] = (uint8_t*)malloc(32);
            if (2 * i + 1 < level_size)
                hash_concat(leaves[2 * i], leaves[2 * i + 1], parents[i]);
            else
                hash_concat(leaves[2 * i], leaves[2 * i], parents[i]);
        }
        pos /= 2;
        for (size_t i = 0; i < level_size; ++i) free(leaves[i]);
        free(leaves);
        leaves = parents;
        level_size = next_level;
    }

    *proof_out = proof;
    return depth;
}

// 验证某个叶子的存在性证明
int verify_merkle_proof(const uint8_t* leaf, size_t index, size_t total, uint8_t** proof, size_t depth, const uint8_t* expected_root) {
    uint8_t computed[32];
    memcpy(computed, leaf, 32);

    for (size_t i = 0; i < depth; ++i) {
        uint8_t temp[32];
        if ((index >> i) & 1)
            hash_concat(proof[i], computed, temp); // sibling || self
        else
            hash_concat(computed, proof[i], temp); // self || sibling
        memcpy(computed, temp, 32);
    }

    return memcmp(computed, expected_root, 32) == 0;
}