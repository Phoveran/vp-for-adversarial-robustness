import torch

def cwloss(fx, y, τ=0., cw1=True, cw2=True, cw3=True):
    '''
    - fx: shape [B, P, C], B means batchsize, P means prompt number and C means class number. P = C or P = 1
    - y: shape [B]
    '''
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    fx = torch.softmax(fx, dim=-1) # confidence scores, sum equal to 1.
    if fx.size(-1) == fx.size(-2):
        class_num = fx.size(-1)
        l = 0 # l: overall loss
        for j in range(class_num): # for each-class samples
            # CW1: max{max_{other_ID /neq y}f_{other_ID}f_{other_ID}(x + δ_{other_ID}) - f_y(x+δ_y), -τ}, for all y.
            other_id = torch.Tensor([k for k in range(class_num) if k != j]).long()
            if cw1:
                fx_y_to_j = fx[y==j] # consider samples with the groundtruth label j
                other_fx = torch.diagonal(fx_y_to_j, dim1=1, dim2=2)[:, other_id] # f_i(x+delta_i) for all i /neq y
                diff_match = other_fx.max(-1).values - fx_y_to_j[:, j, j]
                l += torch.max(diff_match, -τ * torch.ones_like(diff_match)).mean() # CW1
            if cw2:
                # CW2: max{f_j(x + δ_j) - max_{other_ID}f_{other_ID}(x+δ_j), -τ}, for all j.
                fx_y_not_to_j = fx[y!=j]
                diff_non_match = fx_y_not_to_j[:, j, j] - fx_y_not_to_j[:, j, other_id].max(-1).values
                l += torch.max(diff_non_match, -τ * torch.ones_like(diff_non_match)).mean() # CW2
            if cw3:
                # CW3:
                fx_y_to_j = fx[y==j] # consider samples with the groundtruth label j
                other_fx = fx_y_to_j[:, other_id, j]
                diff_match = other_fx.max(-1).values - fx_y_to_j[:, j, j]
                l += torch.max(diff_match, -τ * torch.ones_like(diff_match)).mean() # CW3

    elif fx.size(-2) == 1: # Universal Visual Prompt, where there is only one single VP.
        fx = fx.squeeze()
        match = torch.gather(fx, dim=1, index=y.unsqueeze(1)).squeeze()
        other_idx = torch.Tensor([
            [i for i in range(10) if i != y[j]] for j in range(y.size(0))
        ]).long().to(device)
        non_match_max = torch.gather(fx, dim=1, index=other_idx).max(-1).values
        l = torch.max(non_match_max - match, -τ * torch.ones_like(match)).mean()
    return l


if __name__ == "__main__":
    # Grammar check.
    cwloss(torch.randn(100,10,10), torch.randint(0,10,(100,)))