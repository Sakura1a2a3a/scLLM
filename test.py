
import torch
import numpy as np
import torch.nn.functional as F
import json
from pretrain import SingleCellLanguageModel


with open('./model/gene_token_mapping.json', 'r') as f:
    gene_token_mapping = json.load(f)

with open('./model/expression_mapping.json', 'r') as f:
    expression_mapping = json.load(f)

expression_mapping = {int(k): v for k, v in expression_mapping.items()}




model = SingleCellLanguageModel(
    token_size=len(gene_token_mapping),
    expression_size = len(expression_mapping),
    hidden_size=512
)

model.load_state_dict(torch.load('./model/finetune_model_state_dict.pth'))



selected_token_ids = np.random.choice(np.array(list(gene_token_mapping.values())), 1000, replace=False)

expression = np.full(1000, -1.0)
unmasked_indices = np.random.choice(1000, 313, replace=False)

expression[unmasked_indices] = expression[unmasked_indices] = np.random.randint(0, 5, size=len(unmasked_indices))


print("raw expression", expression)

expression = np.array([expression_mapping[val] for val in expression])
#mask
mask = np.full(1000, True)
mask[unmasked_indices] = False

expression_tensor = torch.LongTensor(expression).unsqueeze(0)
token_ids_tensor = torch.LongTensor(selected_token_ids).unsqueeze(0)

model.eval()
with torch.no_grad():
    predictions = model(token_ids_tensor, expression_tensor).squeeze()

print("prediction", predictions)