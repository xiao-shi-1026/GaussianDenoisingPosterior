import torch
import numpy as np
from torch.linalg import qr

def compute_posterior_pcs(model, y, N=10, K=5, sigma2=25/255**2, c=1e-2, device="cuda"):
    y = y.to(device)

    # 初始化 v₀^(i) ~ N(0, σ²I)
    v = [torch.randn_like(y) * sigma2**0.5 for _ in range(N)]
    v = [v_i.to(device) for v_i in v]

    # μ₁(y) ← 神经网络推理
    with torch.no_grad():
        mu_y = model(y)

    for k in range(K):
        v_new = []
        for i in range(N):
            perturbed = y + c * v[i]
            with torch.no_grad():
                mu_perturbed = model(perturbed)
            diff = mu_perturbed - mu_y
            vk_i = (1 / c) * diff
            v_new.append(vk_i)

        # QR 分解，保持正交性
        V_stack = torch.cat([v_i.unsqueeze(0) for v_i in v_new], dim=0)
        V_flat = V_stack.view(N, -1).T
        Q, _ = torch.linalg.qr(V_flat)
        Q = Q.T.view(N, *y.shape[1:])
        v = [Q[i] for i in range(N)]

    # 计算 λᵢ = (σ² / c)‖μ₁(y + c vᵢ) − μ₁(y)‖
    lambdas = []
    for i in range(N):
        perturbed = y + c * v[i]
        with torch.no_grad():
            mu_perturbed = model(perturbed)
        diff_norm = torch.norm(mu_perturbed - mu_y)
        lambda_i = (sigma2 / c) * diff_norm
        lambdas.append(lambda_i.item())

    return v, lambdas



if __name__ == "__main__":
    from model.UNet import UNet
    import inference as inf
    from data.utils import addnoise
    from data.inpaint import build_inpaint_freeform
    model_path = r"C:\Users\sx119\Desktop\GaussianDenoisingPosterior\outputs\inpainting_unet_2.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inf.load_model(UNet, model_path, device)
    model.to(device)
    img = torch.randn(1, 3, 256, 256).to(device)  # Example input image tensor
    img = img.to(device)
    img = addnoise(img, [5, 5], device)
    class Opt:
        image_size = 256
        device = device
    opt = Opt()
    mask_type = 'freeform1020' 
    inpaint_fn = build_inpaint_freeform(opt, mask_type)
    corrupted_tensor, _ = inpaint_fn(img)
    corrupted_tensor = addnoise(corrupted_tensor, [5, 5], device)
    image_tensor = corrupted_tensor.to(device)

    pcs, lambdas = compute_posterior_pcs(model, image_tensor, N=10, K=5, sigma2=25/255**2, c=1e-2, device=device)
    print("Posterior Principal Components (PCs):")
    for i, pc in enumerate(pcs):
        print(f"PC {i+1}: {pc.shape}")
    print("Lambdas:")
    for i, lambda_i in enumerate(lambdas):
        print(f"Lambda {i+1}: {lambda_i}")
