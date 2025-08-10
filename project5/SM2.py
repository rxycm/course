import os
from gmssl import sm3, func
from typing import Tuple, Optional
import random

# -------------------------
# 优化实现的 SM2（Python）：手写 SM2（基于国密 SM2 曲线参数），SM3 使用 gmssl.sm3。
# 优化点：
#   Jacobian 坐标用于点运算以减少模逆（参考：PDF 中的坐标变换与复杂度分析）。:contentReference[oaicite:1]{index=1}
#   固定基点 G 使用窗口法预计算（windowed fixed-point）以加速 k*G。:contentReference[oaicite:2]{index=2}
#   非固定点使用 NAF（或简单窗口/双倍-加法）减少点加次数。:contentReference[oaicite:3]{index=3}
# 实现 KDF（基于 SM3）
# 加密输出格式遵循 C = C1 || C3 || C2（C1: 64 bytes, C3: 32 bytes, C2: variable）。
# -------------------------

# SM2 曲线参数
p  = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
a  = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
b  = 0x28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93
n  = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
Gx = 0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7
Gy = 0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0
G_affine = (Gx, Gy)

# 工具函数：整数与字节转换
def int_to_bytes(x: int, length: int = 32) -> bytes:
    return x.to_bytes(length, 'big')

def bytes_to_int(b: bytes) -> int:
    return int.from_bytes(b, 'big')

# 基本域运算
def mod_inv(a: int, m: int = p) -> int:
    """模逆，使用 pow 函数计算 a 的模 m 逆"""
    return pow(a % m, -1, m)


# 仿射 / 雅可比 坐标表示
# 雅可比坐标 (X, Y, Z) 表示仿射点 (X/Z^2, Y/Z^3)；无穷点用 Z==0 表示
def is_infinite_j(Pj: Tuple[int,int,int]) -> bool:
    return Pj[2] == 0

def to_jacobian(P: Tuple[int,int]) -> Tuple[int,int,int]:
    x, y = P
    return (x % p, y % p, 1)

def from_jacobian(Pj: Tuple[int,int,int]) -> Optional[Tuple[int,int]]:
    X, Y, Z = Pj
    if Z == 0:
        return None
    Z2 = (Z * Z) % p
    Z3 = (Z2 * Z) % p
    x = (X * mod_inv(Z2, p)) % p
    y = (Y * mod_inv(Z3, p)) % p
    return (x, y)

# Jacobian 点运算（减少模逆）
def point_double_j(P: Tuple[int,int,int]) -> Tuple[int,int,int]:
    X1, Y1, Z1 = P
    if Z1 == 0 or Y1 == 0:
        return (0, 1, 0)  # infinity
    # S = 4*X1*Y1^2
    Y1_sq = (Y1 * Y1) % p
    S = (4 * X1 * Y1_sq) % p
    # M = 3*X1^2 + a*Z1^4
    X1_sq = (X1 * X1) % p
    Z1_sq = (Z1 * Z1) % p
    Z1_4 = (Z1_sq * Z1_sq) % p
    M = (3 * X1_sq + a * Z1_4) % p
    X3 = (M * M - 2 * S) % p
    Y3 = (M * (S - X3) - 8 * (Y1_sq * Y1_sq % p)) % p
    Z3 = (2 * Y1 * Z1) % p
    return (X3, Y3, Z3)

def point_add_j(P: Tuple[int,int,int], Q: Tuple[int,int,int]) -> Tuple[int,int,int]:
    # Handle infinities
    if P[2] == 0:
        return Q
    if Q[2] == 0:
        return P
    X1, Y1, Z1 = P
    X2, Y2, Z2 = Q
    # U1 = X1 * Z2^2, U2 = X2 * Z1^2
    Z1_sq = (Z1 * Z1) % p
    Z2_sq = (Z2 * Z2) % p
    U1 = (X1 * Z2_sq) % p
    U2 = (X2 * Z1_sq) % p
    # S1 = Y1 * Z2^3, S2 = Y2 * Z1^3
    Z1_cu = (Z1_sq * Z1) % p
    Z2_cu = (Z2_sq * Z2) % p
    S1 = (Y1 * Z2_cu) % p
    S2 = (Y2 * Z1_cu) % p
    if U1 == U2:
        if S1 != S2:
            return (0, 1, 0)  # infinity
        return point_double_j(P)
    H = (U2 - U1) % p
    R = (S2 - S1) % p
    H_sq = (H * H) % p
    H_cu = (H_sq * H) % p
    X3 = (R * R - H_cu - 2 * U1 * H_sq) % p
    Y3 = (R * (U1 * H_sq - X3) - S1 * H_cu) % p
    Z3 = (H * Z1 * Z2) % p
    return (X3, Y3, Z3)

# 常用点：仿射G和雅可比形式
GJ = to_jacobian(G_affine)

# 窗口预计算（固定点 G 的预计算）
# windowed fixed-point: precompute [1*G, 2*G, ..., (2^w -1)*G] in Jacobian
# 选择 w=4（表大小 15）可以在内存和速度间做良好权衡（PDF 推荐窗口/预计算）。:contentReference[oaicite:4]{index=4}
WINDOW_SIZE = 4
WINDOW_TABLE_SIZE = (1 << WINDOW_SIZE)  # 16
# we'll store indices 1..(2^w -1) (skip 0)
_fixed_window_table = None

def build_fixed_window_table():
    global _fixed_window_table
    if _fixed_window_table is not None:
        return
    tbl = [None] * WINDOW_TABLE_SIZE
    # compute i*G for i=1..2^w-1
    tbl[0] = (0,1,0)  # unused (0)
    tbl[1] = GJ
    # compute sequentially (this is small cost at startup)
    for i in range(2, WINDOW_TABLE_SIZE):
        # tbl[i] = tbl[i-1] + GJ (using Jacobian add)
        tbl[i] = point_add_j(tbl[i-1], GJ)
    _fixed_window_table = tbl

# 固定点快速点乘 k * G （窗口法）
# 输入 k (int) 输出仿射坐标 (x,y) or None
def fixed_point_mul(k: int) -> Optional[Tuple[int,int]]:
    if k % n == 0:
        return None
    build_fixed_window_table()
    tbl = _fixed_window_table
    # convert to bits and process windows from most significant
    k_bin = bin(k)[2:]
    # pad to multiple of WINDOW_SIZE
    pad_len = (WINDOW_SIZE - (len(k_bin) % WINDOW_SIZE)) % WINDOW_SIZE
    k_bin = ('0' * pad_len) + k_bin
    R = (0,1,0)  # infinity in Jacobian
    for i in range(0, len(k_bin), WINDOW_SIZE):
        # double WINDOW_SIZE times
        for _ in range(WINDOW_SIZE):
            R = point_double_j(R)
        # extract window value
        window_bits = k_bin[i:i+WINDOW_SIZE]
        wval = int(window_bits, 2)
        if wval != 0:
            # add precomputed wval*G
            R = point_add_j(R, tbl[wval])
    return from_jacobian(R)


# 非固定点标量乘（NAF）
# 生成 NAF 表示以减少加法次数
def compute_naf(k: int) -> list:
    """返回 k 的 NAF 编码（低位到高位），每个元素为 0 或奇数（可负）"""
    k = int(k)
    naf = []
    while k > 0:
        if k & 1:
            z = 2 - (k % 4)
            naf.append(z)
            k = k - z
        else:
            naf.append(0)
        k >>= 1
    return naf  # low->high

def mul_point_naf(P: Tuple[int,int], k: int) -> Optional[Tuple[int,int]]:
    if k % n == 0 or P is None:
        return None
    # precompute P and -P in Jacobian
    Pj = to_jacobian(P)
    negPj = (Pj[0], (-Pj[1]) % p, Pj[2])
    naf = compute_naf(k)
    R = (0,1,0)  # infinity in Jacobian
    for digit in reversed(naf):  # high->low
        R = point_double_j(R)
        if digit != 0:
            if digit > 0:
                R = point_add_j(R, Pj)
            else:
                R = point_add_j(R, negPj)
    return from_jacobian(R)



# KDF 使用 SM3 对输入 z 连续哈希 (z || ct) 直至获得 klen 字节
def kdf(z: bytes, klen: int) -> bytes:
    """
    KDF as specified in SM2: 使用 SM3，
    对输入 z 连续哈希 (z || ct) 直至获得 klen 字节。
    """
    ct = 1
    out = b''
    hash_len = 32
    rounds = (klen + hash_len - 1) // hash_len
    for _ in range(rounds):
        msg = z + ct.to_bytes(4, 'big')
        h = sm3.sm3_hash(func.bytes_to_list(msg))
        out += bytes.fromhex(h)
        ct += 1
    return out[:klen]

# SM2 类：封装密钥、加解密、签名、验签
class SM2:
    def __init__(self, d: int = None):
        # 私钥 d（整数），若未提供则随机生成
        if d is None:
            self.priv = int.from_bytes(os.urandom(32), 'big') % n
            if self.priv == 0:
                self.priv = 1
        else:
            self.priv = int(d) % n
        # 公钥点 P = d*G (仿射)
        # 使用固定点加速
        P = fixed_point_mul(self.priv)
        if P is None:
            # shouldn't happen for valid priv
            self.pub = (0, 0)
        else:
            self.pub = P

    def get_priv_hex(self) -> str:
        return hex(self.priv)[2:].zfill(64)

    def get_pub_hex(self) -> str:
        x, y = self.pub
        return hex(x)[2:].zfill(64) + hex(y)[2:].zfill(64)

    # 加密：C = C1 || C3 || C2 ，返回 hex 字符串
    def encrypt(self, msg: bytes, pub_point: Tuple[int,int] = None) -> str:
        if pub_point is None:
            pub_point = self.pub
        if pub_point is None:
            raise ValueError("public key required")
        # loop until kdf != 0
        while True:
            k = int.from_bytes(os.urandom(32), 'big') % n
            if k == 0:
                continue
            C1 = fixed_point_mul(k)
            if C1 is None:
                continue
            # S = k * P (non-fixed point multiply using NAF)
            S = mul_point_naf(pub_point, k)
            if S is None:
                continue
            x2 = int_to_bytes(S[0])
            y2 = int_to_bytes(S[1])
            t = kdf(x2 + y2, len(msg))
            if all(b == 0 for b in t):
                continue
            C2 = bytes([_m ^ _t for _m, _t in zip(msg, t)])
            C3 = bytes.fromhex(sm3.sm3_hash(func.bytes_to_list(x2 + msg + y2)))
            x1 = int_to_bytes(C1[0])
            y1 = int_to_bytes(C1[1])
            return (x1 + y1 + C3 + C2).hex()

    def decrypt(self, cipher_hex: str) -> bytes:
        c = bytes.fromhex(cipher_hex)
        if len(c) < 96:
            raise ValueError("invalid ciphertext length")
        x1 = bytes_to_int(c[:32])
        y1 = bytes_to_int(c[32:64])
        C1 = (x1, y1)
        # compute S = d * C1
        S = mul_point_naf(C1, self.priv)
        if S is None:
            raise ValueError("invalid ciphertext point")
        x2 = int_to_bytes(S[0])
        y2 = int_to_bytes(S[1])
        C3 = c[64:96]
        C2 = c[96:]
        t = kdf(x2 + y2, len(C2))
        if all(b == 0 for b in t):
            raise ValueError("kdf failed")
        m = bytes([_c ^ _t for _c, _t in zip(C2, t)])
        u = bytes.fromhex(sm3.sm3_hash(func.bytes_to_list(x2 + m + y2)))
        if u != C3:
            raise ValueError("Invalid ciphertext")
        return m

    # 签名（按照 SM2 标准，Z = sm3(ENTL||ID||a||b||Gx||Gy||xA||yA)）
    def sign(self, msg: bytes, ID: bytes = b'1234567812345678') -> str:
        # compute Z
        ENTLA = (len(ID) * 8).to_bytes(2, 'big')
        xA = int_to_bytes(self.pub[0])
        yA = int_to_bytes(self.pub[1])
        ZA = sm3.sm3_hash(func.bytes_to_list(ENTLA + ID + int_to_bytes(a) + int_to_bytes(b) + int_to_bytes(Gx) + int_to_bytes(Gy) + xA + yA))
        e = int(sm3.sm3_hash(func.bytes_to_list(bytes.fromhex(ZA) + msg)), 16)
        while True:
            k = int.from_bytes(os.urandom(32), 'big') % n
            if k == 0:
                continue
            P1 = fixed_point_mul(k)
            if P1 is None:
                continue
            x1 = P1[0]
            r = (e + x1) % n
            if r == 0 or r + k == n:
                continue
            s = (mod_inv((1 + self.priv) % n, n) * (k - r * self.priv)) % n
            if s == 0:
                continue
            return hex(r)[2:].zfill(64) + hex(s)[2:].zfill(64)

    def verify(self, msg: bytes, sig_hex: str, pub_point: Tuple[int,int] = None, ID: bytes = b'1234567812345678') -> bool:
        if pub_point is None:
            pub_point = self.pub
        if len(sig_hex) != 128:
            return False
        r = int(sig_hex[:64], 16)
        s = int(sig_hex[64:], 16)
        if not (1 <= r <= n-1 and 1 <= s <= n-1):
            return False
        xA = int_to_bytes(pub_point[0])
        yA = int_to_bytes(pub_point[1])
        ENTLA = (len(ID) * 8).to_bytes(2, 'big')
        ZA = sm3.sm3_hash(func.bytes_to_list(ENTLA + ID + int_to_bytes(a) + int_to_bytes(b) + int_to_bytes(Gx) + int_to_bytes(Gy) + xA + yA))
        e = int(sm3.sm3_hash(func.bytes_to_list(bytes.fromhex(ZA) + msg)), 16)
        t = (r + s) % n
        if t == 0:
            return False
        # compute X = s*G + t*PA
        P1 = fixed_point_mul(s)
        P2 = mul_point_naf(pub_point, t)
        if P1 is None and P2 is None:
            return False
        # add points (convert to jacobian and add)
        P1j = to_jacobian(P1) if P1 is not None else (0,1,0)
        P2j = to_jacobian(P2) if P2 is not None else (0,1,0)
        Xj = point_add_j(P1j, P2j)
        X_aff = from_jacobian(Xj)
        if X_aff is None:
            return False
        x1 = X_aff[0]
        R = (e + x1) % n
        return R == r
    
        # poc验证相关函数
    def sign_with_k(self, msg: bytes, k: int):
        e = int(sm3.sm3_hash(func.bytes_to_list(msg)), 16) % n
        P1 = fixed_point_mul(k)
        x1 = P1[0] % n
        r = (e + x1) % n
        s = (mod_inv((1 + self.priv) % n, n) * ((k - r * self.priv) % n)) % n
        return r, s

    def ecdsa_sign_with_k(self, msg: bytes, k: int):
        e = int(sm3.sm3_hash(func.bytes_to_list(msg)), 16) % n
        P1 = fixed_point_mul(k)
        r = P1[0] % n
        s = (pow(k, -1, n) * ((e + self.priv * r) % n)) % n
        return r, s


# SM2测试（示例）
if __name__ == "__main__":
    print("SM2 Test")
    print("Building fixed window table (G)...")
    build_fixed_window_table()
    print("Creating SM2 keypair...")
    sm2 = SM2()
    print("priv:", sm2.get_priv_hex())
    print("pub :", sm2.get_pub_hex())

    message = b"hello world"
    print("message:", message)

    cipher = sm2.encrypt(message)
    print("cipher(hex):", cipher[:120] + "..." if len(cipher) > 120 else cipher)

    plain = sm2.decrypt(cipher)
    print("decrypted:", plain)

    sig = sm2.sign(message)
    print("signature:", sig)
    print("verify:", sm2.verify(message, sig))

# -------------------------
# poc验证
# -------------------------

# POC: 重复使用 k 导致私钥泄露（同一用户）
def poc_k_reuse_same_user():
    sm2_inst = SM2()
    d = sm2_inst.priv
    print("真实私钥", hex(d))

    k = 0x123456789ABCDEF
    print("复用的 k", hex(k))

    m1 = b"Message 1"
    m2 = b"Message 2"
    r1, s1 = sm2_inst.sign_with_k(m1, k)
    r2, s2 = sm2_inst.sign_with_k(m2, k)

    print("sig1:", hex(r1), hex(s1))
    print("sig2:", hex(r2), hex(s2))

    # 私钥恢复公式
    num = (s1 - s2) % n
    den = (r2 - r1 - s1 + s2) % n
    d_rec = (num * mod_inv(den, n)) % n

    print("恢复私钥", hex(d_rec))
    print("恢复是否正确", d_rec == d)

# POC: 重复使用 k 导致私钥泄露（不同用户）
def poc_k_reuse_cross_users():
    A = SM2()
    B = SM2()
    print("dA:", hex(A.priv))
    print("dB:", hex(B.priv))

    k = 0x2222222222222
    print("共用 k", hex(k))

    rA, sA = A.sign_with_k(b"Alice", k)
    rB, sB = B.sign_with_k(b"Bob", k)

    print("Alice sig:", hex(rA), hex(sA))
    print("Bob   sig:", hex(rB), hex(sB))

    # Alice 已知 dA 和自己的签名 => 恢复 k
    k_calc = (sA + A.priv * ((rA + sA) % n)) % n
    print("计算出的 k:", hex(k_calc), "正确?", k_calc == k)

    # 恢复 dB
    dB_rec = ((k_calc - sB) * mod_inv((rB + sB) % n, n)) % n
    print("恢复的 dB", hex(dB_rec))
    print("恢复是否正确", dB_rec == B.priv)

def sm3_int(msg: bytes) -> int:
    return int(sm3.sm3_hash(func.bytes_to_list(msg)), 16) % n

# POC: SM2 签名与 ECDSA 签名的 k 重用导致私钥泄露
def poc_ecdsa_sm2_crossreuse():
    inst = SM2()
    d = inst.priv
    print("真实私钥", hex(d))

    k = 0x33333333333
    print("共用 k", hex(k))

    r1, s1 = inst.ecdsa_sign_with_k(b"ECDSA-msg", k)
    r2, s2 = inst.sign_with_k(b"SM2-msg", k)

    print("ECDSA-like:", hex(r1), hex(s1))
    print("SM2       :", hex(r2), hex(s2))

    e1 = sm3_int(b"ECDSA-msg")
    num = (s1 * s2 - e1) % n
    den = (r1 - (s1 * ((r2 + s2) % n))) % n
    d_rec = (num * mod_inv(den, n)) % n

    print("恢复私钥", hex(d_rec))
    print("恢复是否正确", d_rec == d)


if __name__ == "__main__":
    print("\nPOC 验证")
    print("1) 重复使用 k 导致私钥泄露（同一用户）")
    poc_k_reuse_same_user()
    print("2) 重复使用 k 导致私钥泄露（不同用户）")
    poc_k_reuse_cross_users()   
    print("3) SM2 签名与 ECDSA 签名的 k 重用导致私钥泄露")
    poc_ecdsa_sm2_crossreuse()


# -------------------------
# 伪造中本聪数字签名 (伪造签名使验证通过)
# -------------------------

def verify_forged_signature(e: int, r: int, s: int, pub_point: tuple) -> bool:
    t = (r + s) % n
    P1 = fixed_point_mul(s)
    P2 = mul_point_naf(pub_point, t)
    if P1 is None and P2 is None:
        return False
    P1j = to_jacobian(P1) if P1 is not None else (0,1,0)
    P2j = to_jacobian(P2) if P2 is not None else (0,1,0)
    Xj = point_add_j(P1j, P2j)
    X_aff = from_jacobian(Xj)
    if X_aff is None:
        return False
    x1 = X_aff[0]
    R = (e + x1) % n
    return R == r

def poc_forge_satoshi_signature():
    satoshi_pub = (0x4A5E1E4BAAB89F3A32518A88B8D7C6F2B9C8D3F1B2F3A4B5C6D7E8F9A0B1C2D3, 0x6B5E1E4BAAB89F3A32518A88B8D7C6F2B9C8D3F1B2F3A4B5C6D7E8F9A0B1C2D4)
    r = 0x12345
    s = 0x67890
    t = (r + s) % n
    P1 = fixed_point_mul(s)
    P2 = mul_point_naf(satoshi_pub, t)
    P1j = to_jacobian(P1) if P1 is not None else (0,1,0)
    P2j = to_jacobian(P2) if P2 is not None else (0,1,0)
    Xj = point_add_j(P1j, P2j)
    X_aff = from_jacobian(Xj)

    x1 = X_aff[0]
    e = (r - x1) % n
    print("伪造签名 r:", hex(r))
    print("伪造签名 s:", hex(s))
    print("伪造哈希 e:", hex(e))
    # 用简化版验证函数验证
    is_valid = verify_forged_signature(e, r, s, satoshi_pub)
    print("验证是否通过:", is_valid)

if __name__ == "__main__":
    print("\n伪造中本聪数字签名")
    poc_forge_satoshi_signature()
