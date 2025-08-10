# Project6

## 一、协议背景与原理

本文档实现的是论文 **Ion et al., IACR ePrint 2019/723** 中 Section 3.1（Figure 2）提出的基于 DDH 假设的 Private Intersection-Sum-with-Cardinality 协议。

协议目标：

- 两方 P1 和 P2 分别持有集合和相关数值，P1 持有标识集合 {v_i}，P2 持有键值对 {(w_j, t_j)}。
- 在不泄露非交集元素的前提下，计算交集的基数和交集中数值的总和。

协议安全性依赖：

- DDH 假设保证哈希到群的标识经随机指数化后无法关联。
- 使用同态加密方案（本实现为 Paillier）来在密文域中进行加法运算，从而计算交集数值和。

* * *

## 二、Paillier 同态加密

Paillier 加密方案是一个经典的公钥同态加密方案，支持明文空间为整数模 n（通常 n=pq）的加法同态。核心思想与算法流程（使用常见的简化选取 g = n + 1 的形式）：

**参数生成 (KeyGen)**

1. 选择两个大素数 p,q，计算 n = p*q，n^2 是模数的平方。
2. 令 λ = lcm(p-1, q-1)。选择 g（通常可选 g = n + 1，能简化计算）。
3. 计算 μ = (L(g^λ mod n^2))^{-1} mod n，其中 L(u) = (u - 1) // n。
4. 公钥 PK = (n, g)，私钥 SK = (λ, μ)。

**加密 (Enc)**

* 对明文 m ∈ Z_n，挑选随机 r ∈ Z_n^*（与 n 互素），计算c = g^m * r^n mod n^2。
* 使用 g = n+1 可化简 g^m mod n^2 = (1 + m*n) mod n^2（有助于加速但实现中仍用幂运算以通用）。

**解密 (Dec)**

* 计算 u = c^λ mod n^2，明文 m = (L(u) * μ) mod n，其中 L(u) = (u-1)//n。

**同态性质**

* 密文乘法对应明文相加：Enc(m1) * Enc(m2) mod n^2 = Enc(m1 + m2 mod n).
* 密文幂运算对应明文乘法：Enc(m)^k mod n^2 = Enc(k*m mod n).

因此，用 Paillier，可以把多个 Paillier 密文直接相乘以得到明文之和的密文；接收方用私钥解密得到和。

---

## 三、协议流程

1. **Round 1（P1→P2）**
   
   - P1 对每个标识 v_i 计算 H(v_i)（哈希到曲线群 G）并进行指数运算 H(v_i)^{k1}。
   - P1 打乱顺序并发送到 P2。

2. **Round 2（P2→P1）**
   
   - P2 将收到的点再指数运算 H(v_i)^{k1k2}。
   - P2 对自己的 w_j 计算 H(w_j)^{k2} 并使用 Paillier 加密对应的 t_j。
   - 将这些成对数据打乱后发送到 P1。

3. **Round 3（P1→P2）**
   
   - P1 对 P2 发来的 H(w_j)^{k2} 再次指数运算得到 H(w_j)^{k1k2}，并与 Round 2 中的集合比对。
   - 对于匹配项，将密文进行同态加法求和。
   - 将求和结果密文重新随机化（加密 0 相乘）后发送给 P2。

4. **输出**
   
   - P2 使用 Paillier 私钥解密得到交集数值和。
   - P1 已在 Round 3 得到交集基数。

---

## 四、相关代码

- **hash_to_scalar / hash_to_point**: 将标识符映射到曲线群 G 的点，满足 H:U→G。
- **Paillier 类**: 实现 KeyGen、Encrypt、Decrypt、同态加法与标量乘法。
- **PIS**: 模拟协议完整流程，内部执行 Round 1~3。
- **pt_to_bytes**: 序列化曲线点用于集合匹配。
- **Example usage**: 构造 P1 和 P2 的测试数据，运行协议并打印交集基数和交集和。

---

## 五、示例结果

![4c76880c-77fc-4a2a-b71b-2b3c437ad332](./pictures/4c76880c-77fc-4a2a-b71b-2b3c437ad332.png)


