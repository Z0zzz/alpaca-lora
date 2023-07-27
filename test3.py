from transformers import LlamaTokenizer
import torch

tokenizer = LlamaTokenizer.from_pretrained("/data/mengke")

tokens = [0, 13866, 338, 385, 15278, 393, 16612, 263, 3414, 29892, 3300,2859, 411, 385, 1881, 393, 8128, 4340, 3030, 29889, 14350, 263, 2933, 393, 7128, 2486, 1614, 2167, 27, 2009, 29889, 13, 13, 2277, 29937, 2799, 4080, 29901, 13, 29954, 5428, 278, 1494, 5665, 310, 2472, 1725, 310, 263, 1734, 322, 967, 6590, 3216, 292, 3800, 310, 278, 3402, 525, 29961, 1742, 29962, 529, 2916, 29896, 29892, 343, 29896, 29892, 921, 29906, 29892, 343, 29906, 29958, 742, 9792, 278, 1959, 518, 1797, 310, 278, 1426, 13, 13, 2277, 29937, 10567, 29901, 13, 5062, 1689, 529, 29906, 29941, 29941, 9892, 29871, 29896, 29900, 29947, 29892, 29871, 29906, 29945, 29906, 29892, 29871, 29896, 29955, 2989, 10202, 13, 19678, 529, 29906, 29945, 29955, 29892, 29871, 29945, 29900, 29906, 29892, 29871, 29906,29955, 29953, 29892, 29871, 29945, 29953, 29955, 10202, 13, 8066, 529, 29906, 29941, 29941, 29892, 2971, 29941, 29941, 29953, 29892, 29871, 29906, 29945, 29906, 29892, 29871, 29941, 29953, 29896, 10202,13, 29898, 30086, 29925, 1150, 4135, 529, 29941, 29896, 29929, 29892, 29871, 29906, 29953, 29906, 29892, 29871, 29941, 29941, 29947, 29892, 29871, 29941, 29900, 29929, 10202, 13, 29879, 545, 529, 29906, 29945, 29955, 29892, 29871, 29941, 29929, 29946, 29892, 29871, 29906, 29955, 29953, 29892, 29871, 29946, 29896, 29955, 10202, 13, 1366, 529, 29906, 29945, 29955, 29892, 29871, 29896, 29896, 29941, 29892, 29871, 29906, 29955, 29953, 29892, 29871, 29896, 29941, 29941, 10202, 13, 3664, 625, 529, 29896, 29906, 29955, 29892, 29871, 29941, 29945, 29929, 29892, 29871, 29896, 29953, 29896, 29892, 29871, 29946, 29906, 29946, 10202, 13, 19678, 529, 29906]
# print(tokenizer.decode(torch.tensor(tokens)))
print(len(tokens))