import os
import cv2
import numpy as np
import pywt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# 指标
def normalized_correlation(a, b):
    a_f = a.astype(np.float64).flatten()
    b_f = b.astype(np.float64).flatten()
    num = np.sum(a_f * b_f)
    den = np.sqrt(np.sum(a_f**2) * np.sum(b_f**2))
    if den == 0:
        return 0.0
    return float(num / den)

# 裁剪/填充
def _crop_or_pad_to_shape(arr, shape):
    h_t, w_t = shape
    arr = np.asarray(arr)
    h, w = arr.shape
    if h >= h_t and w >= w_t:
        return arr[:h_t, :w_t]
    out = np.zeros((h_t, w_t), dtype=arr.dtype)
    out[:h, :w] = arr
    if w < w_t:
        out[:h, w:] = arr[:, -1][:, None]
    if h < h_t:
        out[h:, :] = out[h-1:h, :]
    return out

# 生成水印
def generate_watermark(shape, text='WM'):
    h, w = shape
    wm = np.zeros((h, w), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, min(h, w) // 100)
    scale = 1.0
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    if tw > 0 and th > 0:
        scale = min(max(0.4, (w * 0.7) / tw), max(0.4, (h * 0.7) / th))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    org = ((w - tw) // 2, (h + th) // 2)
    cv2.putText(wm, text, org, font, scale, (255,), thickness, cv2.LINE_AA)
    return wm

# 嵌入水印
def embed_watermark(original_image_bgr, watermark_gray, alpha=0.15, key_path='wm_key.npz'):
    """
    host_bgr: 主图 BGR uint8
    watermark_gray: 灰度水印 uint8 （任意尺寸，会被调整为 LL 大小）
    alpha: 嵌入强度
    key_path: 保存盲提取密钥的路径（npz）
    返回： watermarked_bgr (uint8)
    同时保存 key_path（包含 S_orig, U_w, Vt_w, alpha, ll_shape）
    """
    # 转 YCrCb，取 Y
    host_ycc = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2YCrCb)
    Y = host_ycc[:, :, 0].astype(np.float64)

    # 单层 DWT
    LL, (LH, HL, HH) = pywt.dwt2(Y, 'haar')
    ll_h, ll_w = LL.shape

    # 把 watermark 缩放到 LL 尺寸
    wm_resized = cv2.resize(watermark_gray, (ll_w, ll_h), interpolation=cv2.INTER_LINEAR).astype(np.float64)

    # 对 LL 做 SVD
    U_ll, S_ll, Vt_ll = np.linalg.svd(LL, full_matrices=False)
    # 对水印做 SVD
    U_w, S_w, Vt_w = np.linalg.svd(wm_resized, full_matrices=False)

    # 修改奇异值：相同位置相加（k = min(len(S_ll), len(S_w))）
    k = min(S_ll.shape[0], S_w.shape[0])
    S_ll_mod = S_ll.copy()
    S_ll_mod[:k] = S_ll_mod[:k] + alpha * S_w[:k]

    # 重构 LL
    LL_mod = np.dot(U_ll, np.dot(np.diag(S_ll_mod), Vt_ll))

    # 逆 DWT 并修正尺寸
    Y_mod = pywt.idwt2((LL_mod, (LH, HL, HH)), 'haar')
    Y_mod = _crop_or_pad_to_shape(Y_mod, Y.shape)

    # 放回并转换回 BGR
    host_ycc_mod = host_ycc.copy()
    host_ycc_mod[:, :, 0] = np.clip(Y_mod, 0, 255).astype(np.uint8)
    watermarked_bgr = cv2.cvtColor(host_ycc_mod, cv2.COLOR_YCrCb2BGR)

    # 保存盲密钥（S_orig, U_w, Vt_w, alpha, ll_shape）
    np.savez_compressed(key_path,
                        S_orig=S_ll,
                        U_w=U_w,
                        Vt_w=Vt_w,
                        alpha=np.array([alpha]),
                        ll_shape=np.array([ll_h, ll_w], dtype=np.int32))
    return watermarked_bgr

# 提取水印
def extract_watermark(watermarked_bgr, key_path='wm_key.npz'):
    """
    输入：带水印图 BGR，盲密钥 npz（包含 S_orig, U_w, Vt_w, alpha, ll_shape）
    输出：重构的水印（uint8 灰度, 尺寸 = ll_shape）
    """

    # 读取水印密钥
    data = np.load(key_path, allow_pickle=True)
    S_orig = data['S_orig']
    U_w = data['U_w']
    Vt_w = data['Vt_w']
    alpha = float(data['alpha'].item())
    ll_shape = tuple(data['ll_shape'].astype(int).tolist())

    # DWT 得到 LL
    wm_ycc = cv2.cvtColor(watermarked_bgr, cv2.COLOR_BGR2YCrCb)
    Y_wm = wm_ycc[:, :, 0].astype(np.float64)
    LL_wm, _ = pywt.dwt2(Y_wm, 'haar')

    # 做 SVD
    U_ll_wm, S_ll_wm, Vt_ll_wm = np.linalg.svd(LL_wm, full_matrices=False)

    # 提取水印奇异值： (S_ll_wm - S_orig) / alpha
    k = min(S_ll_wm.shape[0], S_orig.shape[0])
    # 防除零
    denom = alpha if abs(alpha) > 1e-12 else 1e-12
    S_w_extracted = (S_ll_wm[:k] - S_orig[:k]) / denom

    # 用保存的 U_w, Vt_w 重构水印
    k2 = min(S_w_extracted.shape[0], U_w.shape[1], Vt_w.shape[0])
    S_diag = np.zeros((U_w.shape[1], Vt_w.shape[0]), dtype=np.float64)
    S_diag[:k2, :k2] = np.diag(S_w_extracted[:k2])

    wm_recon = np.dot(U_w, np.dot(S_diag, Vt_w))

    # 归一化并裁剪为 ll_shape
    wm_recon = _crop_or_pad_to_shape(wm_recon, ll_shape)
    wm_recon = np.clip(wm_recon, 0, 255)
    wm_uint8 = np.round(wm_recon).astype(np.uint8)
    return wm_uint8

# 鲁棒性测试
def robustness_tests(watermarked_bgr, key_path='wm_key.npz', outdir='robustness', watermark_truth=None):
    os.makedirs(outdir, exist_ok=True)
    h, w = watermarked_bgr.shape[:2]
    results = {}

    # 1) 原样提取（pristine）
    wm_pristine = extract_watermark(watermarked_bgr, key_path)
    cv2.imwrite(os.path.join(outdir, 'extracted_pristine.png'), wm_pristine)
    results['pristine'] = wm_pristine

    # 2) flip horizontal
    flip_h = cv2.flip(watermarked_bgr, 1)
    cv2.imwrite(os.path.join(outdir, 'attack_flip_horizontal.png'), flip_h)
    results['flip_horizontal'] = extract_watermark(flip_h, key_path)

    # 3) translate 50 px right (reflect border)
    M = np.float32([[1, 0, 50], [0, 1, 0]])
    translate = cv2.warpAffine(watermarked_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(os.path.join(outdir, 'attack_translate_50.png'), translate)
    results['translate_50'] = extract_watermark(translate, key_path)

    # 4) crop center 50% and resize back
    crop = watermarked_bgr[h//4:3*h//4, w//4:3*w//4]
    crop_resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(outdir, 'attack_crop_center50.png'), crop_resized)
    results['crop_center50'] = extract_watermark(crop_resized, key_path)

    # 5) contrast increase
    contrast = cv2.convertScaleAbs(watermarked_bgr, alpha=1.5, beta=0)
    cv2.imwrite(os.path.join(outdir, 'attack_contrast_x1.5.png'), contrast)
    results['contrast_x1.5'] = extract_watermark(contrast, key_path)

    # 6) JPEG quality 50
    _, buf = cv2.imencode('.jpg', watermarked_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    jpeg50 = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    cv2.imwrite(os.path.join(outdir, 'attack_jpeg_q50.jpg'), jpeg50)
    results['jpeg_q50'] = extract_watermark(jpeg50, key_path)

    # 7) gaussian noise
    sigma = 10.0
    noise = np.random.normal(0, sigma, watermarked_bgr.shape).astype(np.float32)
    noisy = watermarked_bgr.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(outdir, 'attack_gaussian_sigma10.png'), noisy)
    results['gaussian_sigma10'] = extract_watermark(noisy, key_path)

    # 如果给定 watermark_truth（原始灰度水印），计算指标
    metrics = {}
    if watermark_truth is not None:
        for name, wm_ex in results.items():
            # resize truth to wm_ex shape
            wm_truth_resized = cv2.resize(watermark_truth, (wm_ex.shape[1], wm_ex.shape[0]), interpolation=cv2.INTER_LINEAR)
            nc = normalized_correlation(wm_truth_resized, wm_ex)
            s = None
            try:
                s = float(ssim(wm_truth_resized, wm_ex, data_range=255))
            except Exception:
                s = float('nan')
            p = float(psnr(wm_truth_resized, wm_ex, data_range=255))
            metrics[name] = {'NC': nc, 'SSIM': s, 'PSNR': p}
    # 保存 extracted images
    for name, wm_ex in results.items():
        cv2.imwrite(os.path.join(outdir, f'extracted_{name}.png'), wm_ex)
    return metrics

# 示例
if __name__ == '__main__':
    original_path = 'original_image.jpg'   # 原始图像
    key_path = 'wm_key.npz'                # 提取密钥保存路径
    # 水印图像路径,这里不设置而是使用前面的函数生成水印
    # watermark_path = 'watermark.png'       
    
    # 读取原始图像
    original_bgr = cv2.imread(original_path)

    # 得到 LL 大小
    Y = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    LL, _ = pywt.dwt2(Y, 'haar')
    ll_h, ll_w = LL.shape

    # 生成水印
    watermark = generate_watermark((ll_h, ll_w), text='WM')

    cv2.imwrite('watermark.png', watermark)

    # 嵌入并保存密钥
    alpha = 0.15
    watermarked = embed_watermark(original_bgr, watermark, alpha=alpha, key_path=key_path)
    cv2.imwrite('watermarked_image.png', watermarked)
    print("已保存嵌入水印的图片：watermarked_image.png 以及提取密钥", key_path)

    # 提取水印，并保存结果
    extracted = extract_watermark(watermarked, key_path=key_path)
    cv2.imwrite('extracted_watermark.png', extracted)
    print("已提取水印并保存：extracted_watermark.png")

    # 鲁棒性测试（使用 key 进行提取），并计算指标（使用 watermark truth）
    print("\n鲁棒性测试")
    metrics = robustness_tests(watermarked, key_path=key_path, outdir='robustness', watermark_truth=watermark)
    print("鲁棒性测试指标（NC, SSIM, PSNR）：")
    for name, m in metrics.items():
        print(f"{name}: NC={m['NC']:.4f}, SSIM={m['SSIM']:.4f}, PSNR={m['PSNR']:.2f} dB")

