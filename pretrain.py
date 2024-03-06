import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import scanpy as sc
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import json
from transformer import TransformerEncoder

adata = sc.read_h5ad('./data/sc_human_breast_revised.h5ad')

gene_expression = adata.X.A
gene_names = adata.var_names

gene_token_mapping = {gene: i for i, gene in enumerate(gene_names)}
all_tokens = np.array([gene_token_mapping[gene] for gene in gene_names])

max_expression_value = int(np.max(gene_expression))
expression_mapping = {each: i for i, each in enumerate(range(-1, max_expression_value + 1))}

all_expressions = np.array([expression_mapping.values()])



# with open('gene_token_mapping.json', 'w') as f:
#     json.dump(gene_token_mapping, f)

# with open('expression_mapping.json', 'w') as f:
#     json.dump(expression_mapping, f)



X_train, X_val = train_test_split(gene_expression, test_size=0.2, random_state=0)


class SingleCellDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.all_gene_indices = np.arange(len(all_tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        expression = self.data[index]
        input_tokens = np.array([gene_token_mapping[gene] for gene in gene_names])
        selected_indices= np.random.choice(self.all_gene_indices,500, replace=False)


        expression = expression[selected_indices]
        input_tokens = input_tokens[selected_indices]
        
        mask = np.random.rand(len(expression)) < 0.15
        masked_input_expression = expression.copy()
        masked_input_expression[mask]= -1
        masked_input_expression_indices = np.array([expression_mapping[val] for val in masked_input_expression])


        

        return {
            'masked_input_expression': torch.LongTensor(masked_input_expression_indices),
            'input_tokens': torch.LongTensor(input_tokens),
            'mask': torch.BoolTensor(mask),
            'origin_expression': torch.FloatTensor(expression)
        }


# class SingleCellLanguageModel(nn.Module):
#     def __init__(self, token_size, expression_size, hidden_size):
#         super(SingleCellLanguageModel, self).__init__()

#         self.hidden_size = hidden_size
#         self.token_embedding = nn.Embedding(token_size, hidden_size)
#         self.expression_embedding = nn.Embedding(expression_size, hidden_size)

#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1),
#             num_layers=1
#         )

#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, token_input, expression_input):

#         embedded_token = self.token_embedding(token_input)
        
#         embedded_expression = self.expression_embedding(expression_input)

#         combined_input = embedded_token + embedded_expression

#         transformer_output = self.transformer_encoder(combined_input)

#         output = self.fc(transformer_output)

#         return output

# -------------------------------------------------------------------------------------------------------------

class SingleCellLanguageModel(nn.Module):
    def __init__(self, token_size, expression_size, hidden_size):
        super(SingleCellLanguageModel, self).__init__()

        self.hidden_size = hidden_size
        self.token_embedding = nn.Embedding(token_size, hidden_size)
        self.expression_embedding = nn.Embedding(expression_size, hidden_size)

        self.transformer_encoder = TransformerEncoder(hidden_size = hidden_size, nhead=1, ff_size = 4* hidden_size, num_layers = 1)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, token_input, expression_input, mask= None):

        embedded_token = self.token_embedding(token_input)
        
        embedded_expression = self.expression_embedding(expression_input)

        combined_input = embedded_token + embedded_expression

        transformer_output = self.transformer_encoder(combined_input, mask)

        output = self.fc(transformer_output)

        return output


train_dataset = SingleCellDataset(X_train)
val_dataset = SingleCellDataset(X_val)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)




model = SingleCellLanguageModel(
    token_size=len(gene_token_mapping),
    expression_size = len(expression_mapping),
    hidden_size=8
)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1

if __name__ == "__main__":
    
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        total_epoch_loss = 0 
        for batch in train_dataloader:
            masked_input_expression, input_tokens, mask, origin_expression = (
            batch['masked_input_expression'],
            batch['input_tokens'],
            batch['mask'],
            batch['origin_expression'],
        )

            optimizer.zero_grad()

            gene_expression_predictions = model(input_tokens, masked_input_expression, mask)

            print(gene_expression_predictions.shape)

            predictions = gene_expression_predictions.squeeze()
            target = origin_expression

            masked_predictions = predictions[mask]
            masked_target = target[mask]
        
            loss = criterion(masked_predictions, masked_target)
            print(loss)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_epoch_loss / len(train_dataloader)}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                masked_input_expression, input_tokens, mask, origin_expression = (
                    batch['masked_input_expression'],
                    batch['input_tokens'],
                    batch['mask'],
                    batch['origin_expression'],
                )

                gene_expression_predictions = model(input_tokens, masked_input_expression)

                predictions = gene_expression_predictions.squeeze()
                target = origin_expression

                masked_predictions = predictions[mask]
                masked_target = target[mask]

                val_loss = criterion(masked_predictions, masked_target)
                total_val_loss += val_loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {total_val_loss / len(val_dataloader)}")

# torch.save(model.state_dict(), './model/pretrain_model_state_dict.pth')