import torch.nn as nn
import torch

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    

class Frozenbert(nn.Module):
    def __init__(self, model_config,encoder,num_classes=6):
        super().__init__()

        self.model_config = model_config

        self.encoder = encoder

        self.pool = MeanPooling()
        
        self.fc=nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.Linear(256, num_classes),
        )
        for fclayer in self.fc:
            self._init_weights(fclayer)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, mean=0., std=self.model_config.initializer_range)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.)

        elif isinstance(module, nn.BatchNorm1d):
            nn.init.normal_(module.weight.data, mean=1., std=0.02)
            nn.init.constant_(module.bias.data, val=0.) 


    def feature(self, inputs):
        outputs = self.encoder(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature
    
    def forward(self, inputs):
        feature = self.feature(inputs)
        outputs = self.fc(feature)
        return outputs