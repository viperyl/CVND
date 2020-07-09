import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):

        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM( input_size = embed_size, 
                             hidden_size = hidden_size, 
                             num_layers = num_layers, 
                             dropout = 0.2, 
                             batch_first=True
                           )
        
        self.linear_fc = nn.Linear(hidden_size, vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.embed.weight)
        torch.nn.init.xavier_uniform_(self.linear_fc.weight)

    
    def forward(self, features, captions):
        captions = captions[:, :-1] 

        captions = self.embed(captions)

        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        
        outputs, _ = self.lstm(inputs)
        
        outputs = self.linear_fc(outputs)
        
        return outputs

    
    def sample(self, inputs, states=None, max_len=20):

        outputs = []   
        output_length = 0
        
        while output_length != max_len+1:

            output, states = self.lstm(inputs,states)
           
            output = self.linear_fc(output.squeeze(dim = 1))
            _, predicted_index = torch.max(output, 1)
            
            outputs.append(predicted_index.cpu().numpy()[0].item())
            

            if (predicted_index == 1):
                break
            

            inputs = self.embed(predicted_index)   
            inputs = inputs.unsqueeze(1)
            
            output_length += 1
            
        return outputs
        
         