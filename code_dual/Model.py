import torch.nn as nn
import torch
from torch.autograd import Function

class EncoderLayer(nn.Module):
    def __init__(self, num_input, num_dim):
        super(EncoderLayer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_input, num_dim),
            nn.ReLU()
        )

    def forward(self, input_data):
        coding = self.encoder(input_data)
        return coding

# class ShareEncoder(nn.Module):
#     def __init__(self, num_source, num_target, num_dim_1, num_dim_2):
#         super(ShareEncoder, self).__init__()
#         self.source_encoder = EncoderLayer(num_source, num_dim_1)
#         self.target_encoder = EncoderLayer(num_target, num_dim_1)
#         self.share_encoder = EncoderLayer(num_dim_1, num_dim_2)
        
#     def forward(self, input_source, input_target):
#         coding_source_inter = self.source_encoder(input_source)
#         coding_target_inter = self.target_encoder(input_target)
#         coding_source = self.share_encoder(coding_source_inter)
#         coding_target = self.share_encoder(coding_target_inter)
        # return coding_source, coding_target

class TwoLayerEncoder(nn.Module):
    def __init__(self, num_input, num_dim_1, num_dim_2):
        super(TwoLayerEncoder, self).__init__()
        self.decoder_1 = EncoderLayer(num_input, num_dim_1)
        self.decoder_2 = EncoderLayer(num_dim_1, num_dim_2)
        
    def forward(self, input_data):
        coding_inter = self.decoder_1(input_data)
        coding = self.decoder_2(coding_inter)
        return coding

class Classifier(nn.Module):
    def __init__(self, num_dim_s_2, num_dim_hidden):
        super(Classifier, self).__init__()
        self.encoder = EncoderLayer(num_dim_s_2, num_dim_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(num_dim_hidden, 2),
            nn.Sigmoid()
        )
    def forward(self, input_data, p):
        embeds = self.encoder(input_data)
        embeds_revsers = ReverseLayerF.apply(embeds, p)
        label = self.classifier(embeds_revsers)
        return label

class MF(nn.Module):
    def __init__(self, num_dim_user_2, num_dim_item_2, num_dim_hidden):
        super(MF, self).__init__()
        self.user_encoder = EncoderLayer(num_dim_user_2, num_dim_hidden)
        self.item_encoder = EncoderLayer(num_dim_item_2, num_dim_hidden)

    def forward(self, user_input, item_input):
        user_embeds = self.user_encoder(user_input)
        item_embeds = self.item_encoder(item_input)
        predict_rating = torch.sum(user_embeds * item_embeds, dim=-1)
        return predict_rating

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None

class Dual(nn.Module):
    def __init__(self, num_user_source, num_item_source, num_user_target, num_item_target, \
        num_dim_user_1, num_dim_item_1, num_dim_user_2, num_dim_item_2, num_dim_hidden):
        super(Dual, self).__init__()  
        ##########################################
        # private source encoder
        self.source_encoder_user_p = TwoLayerEncoder(num_item_source, num_dim_user_1, num_dim_user_2)
        self.source_encoder_item_p = TwoLayerEncoder(num_user_source, num_dim_item_1, num_dim_item_2)

        # private target encoder
        self.target_encoder_user_p = TwoLayerEncoder(num_item_target, num_dim_user_1, num_dim_user_2)
        self.target_encoder_item_p = TwoLayerEncoder(num_user_target, num_dim_item_1, num_dim_item_2)

        ##########################################
        # share user encoder
        self.source_encoder_user_s = EncoderLayer(num_item_source, num_dim_user_1)
        self.target_encoder_user_s = EncoderLayer(num_item_target, num_dim_user_1)


        # share item encoder 
        self.source_encoder_item_s = EncoderLayer(num_user_source, num_dim_item_1)
        self.target_encoder_item_s = EncoderLayer(num_user_target, num_dim_item_1)

        # share encoder
        self.encoder_user_s  = EncoderLayer(num_dim_user_1, num_dim_user_2)
        self.encoder_item_s  = EncoderLayer(num_dim_item_1, num_dim_item_2)

        # classifier
        self.user_classifier = Classifier(num_dim_user_2, num_dim_hidden)
        self.item_classifier = Classifier(num_dim_item_2, num_dim_hidden)


        ##########################################
        # source decoder
        self.source_decoder_user = TwoLayerEncoder(num_dim_user_2, num_dim_user_1, num_item_source)
        self.source_decoder_item = TwoLayerEncoder(num_dim_item_2, num_dim_item_1, num_user_source)

        # source decoder
        self.target_decoder_user = TwoLayerEncoder(num_dim_user_2, num_dim_user_1, num_item_target)
        self.target_decoder_item = TwoLayerEncoder(num_dim_item_2, num_dim_item_1, num_user_target)
 
        ##########################################
        # target MF
        self.MF = MF(num_dim_user_2, num_dim_item_2, num_dim_hidden)      
        

        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean = 0, std = 0.01)
                m.bias.data.zero_()

    
    def forward(self, user_input, item_input, p=0.0, source_scheme = 'target', rec_scheme = 'all', predict_scheme = 'recon'):
        if source_scheme == 'source':
            user_embeds_p = self.source_encoder_user_p(user_input)
            item_embeds_p = self.source_encoder_item_p(item_input)
            user_embeds_s1 = self.source_encoder_user_s(user_input)
            item_embeds_s1 = self.source_encoder_item_s(item_input)
        elif source_scheme == 'target':
            user_embeds_p = self.target_encoder_user_p(user_input)
            item_embeds_p = self.target_encoder_item_p(item_input)
            user_embeds_s1 = self.target_encoder_user_s(user_input)
            item_embeds_s1 = self.target_encoder_item_s(item_input)
               
        user_embeds_s = self.encoder_user_s(user_embeds_s1)
        item_embeds_s = self.encoder_item_s(item_embeds_s1)        
        user_label = self.user_classifier(user_embeds_s, p)
        item_label = self.item_classifier(item_embeds_s, p)

        if rec_scheme == 'share':
            user_code = user_embeds_s
            item_code = item_embeds_s
        elif rec_scheme == 'private':
            user_code = user_embeds_p
            item_code = item_embeds_p
        elif rec_scheme == 'all':
            user_code = user_embeds_s + user_embeds_p
            item_code = item_embeds_s + item_embeds_p

        if predict_scheme == 'MF':            
            rating_predict = self.MF(user_code, item_code)
        elif predict_scheme == 'recon':
            if source_scheme == 'source':
                predict_user = self.source_decoder_user(user_code)
                predict_item = self.source_decoder_item(item_code)
            elif source_scheme == 'target':
                predict_user = self.target_decoder_user(user_code)
                predict_item = self.target_decoder_item(item_code)
            rating_predict = [predict_user, predict_item]
            

        embeds = [user_embeds_p, item_embeds_p, user_embeds_s, item_embeds_s]
        results = [user_label, item_label, rating_predict]
        return embeds, results
    

# class item_Dual(nn.Module):
#     def __init__(self, num_user_source, num_user_target, num_dim_item_1, num_dim_item_2, num_dim_hidden):
#         super(item_Dual, self).__init__()
#         self.source_encoder_item_p = TwoLayerEncoder(num_user_source, num_dim_item_1, num_dim_item_2)
#         self.target_encoder_item_p = TwoLayerEncoder(num_user_target, num_dim_item_1, num_dim_item_2)
#         # share item encoder 
#         self.source_encoder_item_s = EncoderLayer(num_user_source, num_dim_item_1)
#         self.target_encoder_item_s = EncoderLayer(num_user_target, num_dim_item_1)
#         self.encoder_item_s  = EncoderLayer(num_dim_item_1, num_dim_item_2)
#         self.item_classifier = Classifier(num_dim_item_2, num_dim_hidden)

#         ##########################################
#         # decoder
#         self.source_decoder_item = TwoLayerEncoder(num_dim_item_2, num_dim_item_1, num_user_source)
#         self.target_decoder_item = TwoLayerEncoder(num_dim_item_2, num_dim_item_1, num_user_target)
        
#         # initial model
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean = 0, std = 0.01)
#                 m.bias.data.zero_()

#     def forward(self, item_input, p=0.0, source_scheme = 'target'):
#         if source_scheme == 'source':
#             item_embeds_p = self.source_encoder_item_p(item_input)
#             item_embeds_s1 = self.source_encoder_item_s(item_input)
#         elif source_scheme == 'target':
#             item_embeds_p = self.target_encoder_item_p(item_input)
#             item_embeds_s1 = self.target_encoder_item_s(item_input)        
        
#         item_embeds_s = self.encoder_item_s(item_embeds_s1)        
#         item_label = self.item_classifier(item_embeds_s, p)

 
#         item_code = item_embeds_s + item_embeds_p

#         if source_scheme == 'source':
#             predict_item = self.source_decoder_item(item_code)
#         elif source_scheme == 'target':
#             predict_item = self.target_decoder_item(item_code)
            
#         embeds = [item_embeds_p, item_code]
#         results = [item_label, predict_item]
#         return embeds, results
            

class user_Dual(nn.Module):
    def __init__(self, num_item_source, num_item_target, num_dim_user_1, num_dim_user_2, num_dim_hidden):
        super(user_Dual, self).__init__()  
        ##########################################
        # private source encoder
        self.source_encoder_user_p = TwoLayerEncoder(num_item_source, num_dim_user_1, num_dim_user_2)
        # private target encoder
        self.target_encoder_user_p = TwoLayerEncoder(num_item_target, num_dim_user_1, num_dim_user_2)
        ##########################################
        # share user encoder
        self.source_encoder_user_s = EncoderLayer(num_item_source, num_dim_user_1)
        self.target_encoder_user_s = EncoderLayer(num_item_target, num_dim_user_1)
        # share encoder
        self.encoder_user_s  = EncoderLayer(num_dim_user_1, num_dim_user_2)
        # classifier
        self.user_classifier = Classifier(num_dim_user_2, num_dim_hidden)
        ##########################################
        # decoder
        self.source_decoder_user = TwoLayerEncoder(num_dim_user_2, num_dim_user_1, num_item_source)
        self.target_decoder_user = TwoLayerEncoder(num_dim_user_2, num_dim_user_1, num_item_target)
            

        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean = 0, std = 0.01)
                m.bias.data.zero_()

    
    def forward(self, user_input, p=0.0, source_scheme = 'target'):
        if source_scheme == 'source':
            user_embeds_p = self.source_encoder_user_p(user_input)
            user_embeds_s1 = self.source_encoder_user_s(user_input)
        elif source_scheme == 'target':
            user_embeds_p = self.target_encoder_user_p(user_input)          
            user_embeds_s1 = self.target_encoder_user_s(user_input)  
               
        user_embeds_s = self.encoder_user_s(user_embeds_s1)       
        user_label = self.user_classifier(user_embeds_s, p)


        user_code = user_embeds_s + user_embeds_p
        
        if source_scheme == 'source':
            predict_user = self.source_decoder_user(user_code)  
        elif source_scheme == 'target':
            predict_user = self.target_decoder_user(user_code)


        embeds = [user_embeds_p, user_code]
        results = [user_label, predict_user]
        return embeds, results


class one_side_Dual(nn.Module):
    def __init__(self, num_user_source, num_item_source, num_user_target, num_item_target, \
        num_dim_user_1, num_dim_item_1, num_dim_user_2, num_dim_item_2, num_dim_hidden):
        super(one_side_Dual, self).__init__()  
        ##########################################
        # private source encoder
        self.source_encoder_user_p = TwoLayerEncoder(num_item_source, num_dim_user_1, num_dim_user_2)
        self.source_encoder_item_p = TwoLayerEncoder(num_user_source, num_dim_item_1, num_dim_item_2)

        # private target encoder
        self.target_encoder_user_p = TwoLayerEncoder(num_item_target, num_dim_user_1, num_dim_user_2)
        self.target_encoder_item_p = TwoLayerEncoder(num_user_target, num_dim_item_1, num_dim_item_2)

        ##########################################
        # share user encoder
        self.source_encoder_user_s = EncoderLayer(num_item_source, num_dim_user_1)
        self.target_encoder_user_s = EncoderLayer(num_item_target, num_dim_user_1)


        # share item encoder 
        self.source_encoder_item_s = EncoderLayer(num_user_source, num_dim_item_1)
        self.target_encoder_item_s = EncoderLayer(num_user_target, num_dim_item_1)

        # share encoder
        self.encoder_user_s  = EncoderLayer(num_dim_user_1, num_dim_user_2)
        self.encoder_item_s  = EncoderLayer(num_dim_item_1, num_dim_item_2)

        # classifier
        self.user_classifier = Classifier(num_dim_user_2, num_dim_hidden)
        self.item_classifier = Classifier(num_dim_item_2, num_dim_hidden)


        ##########################################
        # source decoder
        self.source_decoder_user = TwoLayerEncoder(num_dim_user_2, num_dim_user_1, num_item_source)
        self.source_decoder_item = TwoLayerEncoder(num_dim_item_2, num_dim_item_1, num_user_source)

        # source decoder
        self.target_decoder_user = TwoLayerEncoder(num_dim_user_2, num_dim_user_1, num_item_target)
        self.target_decoder_item = TwoLayerEncoder(num_dim_item_2, num_dim_item_1, num_user_target)
 
        ##########################################
        # target MF
        self.MF = MF(num_dim_user_2, num_dim_item_2, num_dim_hidden)      
        

        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean = 0, std = 0.01)
                m.bias.data.zero_()

    
    def forward(self, user_input, item_input, p=0.0, source_scheme = 'target', share_scheme = 'user'):
        if source_scheme == 'source':
            user_embeds_p = self.source_encoder_user_p(user_input)
            item_embeds_p = self.source_encoder_item_p(item_input)
            user_embeds_s1 = self.source_encoder_user_s(user_input)
            item_embeds_s1 = self.source_encoder_item_s(item_input)
        elif source_scheme == 'target':
            user_embeds_p = self.target_encoder_user_p(user_input)
            item_embeds_p = self.target_encoder_item_p(item_input)
            user_embeds_s1 = self.target_encoder_user_s(user_input)
            item_embeds_s1 = self.target_encoder_item_s(item_input)
               
        user_embeds_s = self.encoder_user_s(user_embeds_s1)
        item_embeds_s = self.encoder_item_s(item_embeds_s1)        
        user_label = self.user_classifier(user_embeds_s, p)
        item_label = self.item_classifier(item_embeds_s, p)

        if share_scheme == 'user':
            user_code = user_embeds_s + user_embeds_p
            item_code = item_embeds_p
        elif share_scheme == 'item':
            user_code = user_embeds_p
            item_code = item_embeds_s + item_embeds_p
    
                
        rating_predict = self.MF(user_code, item_code)           

        embeds = [user_embeds_p, item_embeds_p, user_code, item_code]
        results = [user_label, item_label, rating_predict]
        return embeds, results