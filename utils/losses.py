import torch.nn as nn 
import torch 

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def cosine_similarity(self, emb1, emb2):
        cos =  nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(emb1, emb2)

    def forward(self, anchor, pos, neg):
        pos_distance = self.cosine_similarity(anchor, pos)
        neg_distance = self.cosine_similarity(anchor, neg)
        losses = torch.relu(pos_distance - neg_distance + self.margin)
        return losses.mean()

if __name__ == "__main__":
    input1 = torch.randn(3, 10)
    input2 = torch.randn(3,10)
    anchor = torch.randn(3, 10)

    triplet_loss = TripletLoss()
    loss = triplet_loss(anchor, input1, input2)
    print(loss)