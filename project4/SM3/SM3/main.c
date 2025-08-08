#define _CRT_SECURE_NO_WARNINGS

#include"sm3.h"
#include<stdio.h>
#include <stdlib.h>

#define LEAF_COUNT 100000

int main() {
	//sm3
	const uint8_t* data = "123456789";

	uint8_t hash[32] = {0};
	sm3_hash(data, strlen(data), hash);

	printf("sm3_hash:\n");
	for (size_t i = 0; i < 32; i++)
	{
		printf("0x%02x ", hash[i]);
	}

	//length_extension_attack
	printf("\n\nlength_extension_attack:\n");
	
	const uint8_t* append = "0";
	uint8_t new_hash[32] = { 0 };
	sm3_length_extension_attack(hash, strlen(data), append, strlen(append), new_hash);
	uint8_t msg_hash[32] = { 0 };
	//message = data || pad || append
	uint8_t* pad = NULL;
	size_t pad_len = sm3_padding(data, strlen(data), &pad);
	size_t forged_len = pad_len + strlen(append);
	uint8_t* forged_msg = (uint8_t*)malloc(forged_len);
	memcpy(forged_msg, pad, pad_len);
	memcpy(forged_msg + pad_len, append, strlen(append));
	free(pad);
	printf("forged message:");
	for (size_t i = 0; i < forged_len; i++)
	{
		printf("%02x ", forged_msg[i]);
	};
	printf("\n");

	sm3_hash(forged_msg, forged_len, msg_hash);
	printf("message hash:");
	for (size_t i = 0; i < 32; i++)
	{
		printf("0x%02x ", msg_hash[i]);
	}
	printf("\nattack result:");
	for (size_t i = 0; i < 32; i++)
	{
		printf("0x%02x ", new_hash[i]);
	}
	printf("\n");
	free(forged_msg);

	//merkle tree
	printf("\n构建包含 %d 个叶子的 SM3 Merkle 树\n", LEAF_COUNT);

	// 1. 生成叶子：leaf[i] = sm3("leaf-i")
	uint8_t** leaves = (uint8_t**)malloc(sizeof(uint8_t*) * LEAF_COUNT);
	char buf[32];
	for (int i = 0; i < LEAF_COUNT; ++i) {
		sprintf(buf, "leaf-%d", i);
		leaves[i] = (uint8_t*)malloc(32);
		sm3_hash((uint8_t*)buf, strlen(buf), leaves[i]);
	}

	// 2. 构建 Merkle 树
	MerkleTree* tree = build_merkle_tree(leaves, LEAF_COUNT);

	// 3. 获取随机一个叶子的存在性证明
	size_t test_index = rand() % LEAF_COUNT;
	uint8_t** proof = NULL;
	uint8_t* leaf = (uint8_t*)malloc(32);
	memcpy(leaf, leaves[test_index], 32);

	// 复制叶子
	uint8_t** leaf_copy = (uint8_t**)malloc(LEAF_COUNT * sizeof(uint8_t*));
	for (int i = 0; i < LEAF_COUNT; ++i) {
		leaf_copy[i] = (uint8_t*)malloc(32);
		memcpy(leaf_copy[i], leaves[i], 32);
	}

	size_t depth = get_merkle_proof(leaf_copy, LEAF_COUNT, test_index, &proof);

	// 4. 验证该叶子是否存在
	printf("存在性证明:\n");
	printf("验证 index=%zu 的叶子是否存在于 Merkle 树中...\n", test_index);
	int valid = verify_merkle_proof(leaf, test_index, LEAF_COUNT, proof, depth, tree->root);
	printf("验证结果：%s\n", valid ? "存在" : "不存在");

	// 5. 模拟不存在性证明（错误 index 或错误数据）
	uint8_t fake_leaf[32];
	sm3_hash((const uint8_t*)"non-existing-leaf", 18, fake_leaf);
	printf("验证一个不存在的叶子（non-existing-leaf）...\n");
	int fake_valid = verify_merkle_proof(fake_leaf, 123456, LEAF_COUNT, proof, depth, tree->root);
	printf("验证结果：%s\n", fake_valid ? "存在" : "不存在");

	return 0;
}
