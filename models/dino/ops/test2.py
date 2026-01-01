import torch
from functions.ms_deform_attn_func import MSDeformAttnFunction
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
# tiny forward
value = torch.rand(1, 10, 2, 4).cuda()
shapes = torch.as_tensor([(2,5)], dtype=torch.long).cuda()
level_start_index = torch.tensor([0], dtype=torch.long).cuda()
sampling_locations = torch.rand(1, 3, 2, 1, 2, 2).cuda()
attention_weights = torch.rand(1, 3, 2, 1, 2).cuda()
out = MSDeformAttnFunction.apply(value, shapes, level_start_index, sampling_locations, attention_weights, 2)
print("forward OK", out.shape)