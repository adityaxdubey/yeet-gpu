import torch

def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    torch.add(A, B, out=C)
    return C

if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N=1000
    A=torch.randn(N, N, device=device)
    B=torch.randn(N, N, device=device)
    C=torch.zeros(N, N, device=device)
    solve(A,B,C, N)
    print(C[:3,:3])
