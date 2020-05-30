import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """Encoder Netwok - ResNet50 striped of the last layer raplased with the custom """
    
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

        
    def forward(self, images):
        encoding_vector = self.resnet(images)
        encoding_vector = encoding_vector.view(encoding_vector.size(0), -1)
        encoding_vector = self.embed(encoding_vector)
        return encoding_vector


    
class DecoderRNN(nn.Module):
    """ Decoder Netowrk - Network design to interpret the encoded image data and produce a sentance"""

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_dim = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size))
   

    def forward(self, features, captions):
        """ Forward pass of the RNN network"""
        cap_embedding = self.embed(captions[:,:-1])                      
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), 1)
        lstm_out, self.hidden = self.lstm(embeddings)                              
        outputs = self.linear(lstm_out)
        return outputs


    def sample(self, inputs, hidden=None, max_len=20):
        """ Takes pre-processed image tensor and returns predicted sentence."""
        res = []
        
        for i in range(max_len):
            outputs, hidden = self.lstm(inputs, hidden)
            outputs = self.linear(outputs.squeeze(1))
            target_index = outputs.max(1)[1]
            res.append(target_index.item())
            inputs = self.embed(target_index).unsqueeze(1)
        return res
