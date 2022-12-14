import torch
import torch.nn as nn
import math
import collections
import pickle
import time

print("__name__ ",__name__)

# USER INPUT STRING MANIPULATION

def standardizeString(string, is_string_target):

    def _standardizeString(string):

        space_characters = ['\u202f', '\u2009','\xa0'];
        special_characters = '«»&~"#\'{([-|`_\\^@)]=}+¨£$¤%µ,?;.:!§*<>';
        numbers = '0123456789';
        
        ## Remove space characters
        for i in range(len(space_characters)):
            if space_characters[i] in string:
                string = string.replace(space_characters[i], ' ');
            
        len_string, _string = len(string), '';
        for i, char in enumerate(string):
            
            ## Handle special characters and numbers
            if char in special_characters or char in numbers:
                left_space, right_space = '', '';
                
                if i > 0 and string[i-1] != ' ':
                    left_space = ' ';

                if i+1 < len_string and string[i+1] != ' ' and string[i+1] not in special_characters and string[i+1] not in numbers:
                    right_space = ' ';
                
                _string += left_space + char + right_space;

            else:
                _string += char;

        return _string.lower().split(' ');


    _string = [s.strip() for s in string.split(' ') if s.strip() != ''];

    output = [];
    for i, _str in enumerate(_string):
        output += _standardizeString(_str);
        if i+1 < len(_string):
            output += ['<space>'];

    if is_string_target:
        output = ['<bos>'] + output;

    return output + ['<eos>'];     


def dataLoader(batch_size, shuffle, *tensors):
    TD = torch.utils.data.TensorDataset(*tensors);
    return torch.utils.data.DataLoader(TD, batch_size, shuffle);


def sequencesLen(dataset_examples):
    sequences_len = [];
    
    for i in range(len(dataset_examples)):
        sequences_len.append(len(dataset_examples[i]));
        
    return torch.tensor(sequences_len);

# VOCAB

class Vocab:
    def __init__(self, dataset2d):
        self.token_to_idx = {};
        self.idx_to_token = [];
        self.initVocab(dataset2d);
        
    def initVocab(self, dataset2d):
        token_freq = collections.Counter(
            [dataset2d[i][j] for i in range(len(dataset2d)) for j in range(len(dataset2d[i]))]);
        token_freq = token_freq.most_common();
  
        for i in range(len(token_freq)):
            self.token_to_idx[token_freq[i][0]] = i;
            self.idx_to_token.append(token_freq[i][0]);
             
    def tokenToIdx(self, dataset2d):
        for i in range(len(dataset2d)):
            dataset2d_irow = [];
            for j in range(len(dataset2d[i])):
                current_token = dataset2d[i][j];

                if current_token not in self.idx_to_token:
                    dataset2d_irow.append(self.token_to_idx['<special_begin>'])
                    for token in current_token:
                        if token not in self.idx_to_token:
                            dataset2d_irow.append(self.token_to_idx['<ukn>']);
                        else:
                            dataset2d_irow.append(self.token_to_idx[token]);
                    dataset2d_irow.append(self.token_to_idx['<special_end>'])
                else:
                    dataset2d_irow.append(self.token_to_idx[current_token]);

            dataset2d[i] = dataset2d_irow;
                    
        return torch.tensor(dataset2d);

    def idxToToken(self, dataset2d):
        dataset2d = dataset2d.tolist();
        
        for i in range(len(dataset2d)):
            for j in range(len(dataset2d[i])):
                dataset2d[i][j] = self.idx_to_token[dataset2d[i][j]];
        return dataset2d;

    def expandVocab(self, dataset2d):
        token_freq = collections.Counter(
            [dataset2d[i][j] for i in range(len(dataset2d)) for j in range(len(dataset2d[i]))]);
        token_freq = token_freq.most_common();
  
        for i in range(len(token_freq)):
            if token_freq[i][0] not in self.idx_to_token:
                self.token_to_idx[token_freq[i][0]] = len(self.idx_to_token);
                self.idx_to_token.append(token_freq[i][0]);
    
    def __len__(self):
        return len(self.idx_to_token);

# MODEL

def maskedSoftmax(QK, source_seq_len, mask):
    # QK.shape = (batch_size, num_steps, num_steps)
    
    QK_shape = QK.shape;
    
    if mask is True:
        mask_to_apply = ~(torch.arange(0,QK_shape[1])[None,:] < torch.arange(1,QK_shape[1]+1)[:,None]);
        mask_to_apply = mask_to_apply.unsqueeze(dim=0).repeat(QK_shape[0],1,1);
        
        QK[mask_to_apply] = -1e6;
    
    if source_seq_len is not None:
        steps = torch.arange(1, QK_shape[1]+1).unsqueeze(dim=0).repeat(QK_shape[1],1).unsqueeze(dim=0).repeat(QK_shape[0], 1, 1);
        valid_len = source_seq_len.unsqueeze(dim=1).unsqueeze(dim=1).repeat_interleave(repeats=QK_shape[1], dim=1);
        padding_mask = steps > valid_len;
        
        QK[padding_mask] = -1e6;
    
    return nn.functional.softmax(QK, dim=-1);

def scaledDotProductAttention(Q, K, V, dk, source_seq_len, mask):
    QK = torch.bmm(Q,K.transpose(1,2)) / math.sqrt(dk);
    
    return torch.bmm(maskedSoftmax(QK, source_seq_len, mask), V);

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dk, dv, dmodel):
        super().__init__();
        
        self.num_heads = num_heads;
        self.dk = dk;
        
        self.weights_params = nn.ModuleList();
        for i in range(num_heads):
            WQi = nn.Linear(dmodel,dk);
            WKi = nn.Linear(dmodel,dk);
            WVi = nn.Linear(dmodel,dv);
            
            weights = nn.ModuleList([WQi,WKi,WVi]);
            self.weights_params.append(weights);
            
        self.WO = nn.Linear(num_heads*dv,dmodel);
    
    def forward(self, queries, keys, values, source_seq_len=None, mask=False):
        # (queries|keys|values).shape = (batch_size, num_steps, dmodel)
        
        heads = [];
        
        for i in range(self.num_heads):
            WQi, WKi, WVi = self.weights_params[i];
            
            # ith head shape = (batch_size, num_steps, dv)
            heads.append(
                scaledDotProductAttention(WQi(queries), WKi(keys), WVi(values), self.dk, source_seq_len, mask));
        
        # heads.shape = (batch_size, num_steps, num_heads*dv)
        heads = torch.cat(heads, dim=-1);
        
        return self.WO(heads);

class FFN(nn.Module):
    def __init__(self, dmodel, dff):
        super().__init__();
        
        self.W1 = nn.Linear(dmodel,dff);
        self.W2 = nn.Linear(dff,dmodel);
        self.relu = nn.ReLU();
        
    def forward(self, X):
        # X.shape = (batch_size, num_steps, dmodel)
        
        return self.W2(self.relu(self.W1(X)));

class AddandNorm(nn.Module):
    def __init__(self, dmodel, dropout=0):
        super().__init__();
        
        self.LN = nn.LayerNorm(dmodel);
        self.dropout = nn.Dropout(dropout);
    
    def forward(self, X, Y):
        # (X|Y).shape = (batch_size, num_steps, dmodel)

        return self.LN(X + self.dropout(Y));

class PositionalEncoding(nn.Module):
    def __init__(self, dmodel, dropout, max_seq_len=1000):
        super().__init__();
        
        # It is possible that dmodel is odd but this makes the code more complex without adding value.
        assert dmodel % 2 == 0, "dmodel must be even";
        
        self.dropout = nn.Dropout();
        
        # t.shape = (max_seq_len, dmodel/2)
        t = torch.arange(0,max_seq_len).unsqueeze(dim=1).repeat_interleave(repeats=int(dmodel/2),dim=1);
        # w.shape = (max_seq_len, dmodel/2)
        wk = 1/torch.pow(10000, torch.arange(0,dmodel,step=2)/dmodel).unsqueeze(dim=0);
        wk = wk.repeat_interleave(repeats=max_seq_len,dim=0);
        
        # pos_encoding.shape = (max_seq_len, dmodel)
        self.pos_encoding = torch.zeros(max_seq_len, dmodel);
        self.pos_encoding[:,0::2] = torch.sin(wk*t);
        self.pos_encoding[:,1::2] = torch.cos(wk*t);

    def forward(self, X):
        # X.shape = (batch_size, num_steps, dmodel)
        X_shape = X.shape;
        
        pos_encoding = self.pos_encoding[:X_shape[1],:].unsqueeze(dim=0).repeat(X.shape[0],1,1);
        
        return self.dropout(pos_encoding + X);

class EncoderBlock(nn.Module):
    def __init__(self, num_heads, dmodel, dk, dv, dff, dropout):
        super().__init__();
        
        self.MHA = MultiHeadAttention(num_heads, dk, dv, dmodel);
        self.AAN = AddandNorm(dmodel, dropout);
        self.FFN = FFN(dmodel, dff);

    def forward(self, X, source_seq_len):
        # sli_out.shape = (batch_size, number of steps in src_X, dmodel)
        # sli stands for the ith sublayer of the encoder block.
        
        sl1_out = self.MHA(X, X, X, source_seq_len);
        sl1_out = self.AAN(X, sl1_out);
        
        sl2_out = self.FFN(sl1_out)
        sl2_out = self.AAN(sl1_out, sl2_out);
        
        return sl2_out;

class Encoder(nn.Module):
    def __init__(self, num_blocks, vocab_size, num_heads, dmodel, dk, dv, dff, dropout, max_seq_len=1000):
        super().__init__();
        
        self.num_blocks = num_blocks;
        self.embedding = nn.Embedding(vocab_size, dmodel);
        self.pencoding = PositionalEncoding(dmodel, dropout, max_seq_len);
        
        self.encoder_blocks = nn.ModuleList();
        for i in range(num_blocks):
            self.encoder_blocks.append(EncoderBlock(num_heads, dmodel, dk, dv, dff, dropout));

    def forward(self, src_X, source_seq_len_train):
        # X.shape = (batch_size, number of steps in src_X, dmodel)
        X = self.pencoding(self.embedding(src_X));
            
        for i in range(self.num_blocks):
            X = self.encoder_blocks[i](X, source_seq_len_train);
            
        return X;

class DecoderBlock(nn.Module):
    def __init__(self, num_heads, dmodel, dk, dv, dff, dropout):
        super().__init__();
        
        self.MHA1 = MultiHeadAttention(num_heads, dk, dv, dmodel);
        self.MHA2 = MultiHeadAttention(num_heads, dk, dv, dmodel);
        self.AAN = AddandNorm(dmodel, dropout);
        self.FFN = FFN(dmodel, dff);
        
    def forward(self, X, enc_output, mask=False):
        # X.shape = (batch_size, number of steps in bos_X, dmodel)
        # enc_output.shape = (batch_size, number of steps in src_X, dmodel)
        
        # sli_out.shape = (batch_size, number of steps in bos_X, dmodel)
        # sli stands for the ith sublayer of the decoder block.
        
        sl1_out = self.MHA1(X, X, X, None, mask);
        sl1_out = self.AAN(X, sl1_out);
        
        sl2_out = self.MHA2(sl1_out, enc_output, enc_output);
        sl2_out = self.AAN(sl1_out, sl2_out);
        
        sl3_out = self.FFN(sl2_out);
        sl3_out = self.AAN(sl2_out, sl3_out);
        
        return sl3_out; 

class Decoder(nn.Module):
    def __init__(self, num_blocks, vocab_size, num_heads, dmodel, dk, dv, dff, dropout, max_seq_len=1000):
        super().__init__();
        
        self.num_blocks = num_blocks;
        self.embedding = nn.Embedding(vocab_size, dmodel);
        self.pencoding = PositionalEncoding(dmodel, dropout, max_seq_len);
        
        self.W_out = nn.Linear(dmodel, vocab_size);
        
        self.decoder_blocks = nn.ModuleList();
        for i in range(num_blocks):
            self.decoder_blocks.append(DecoderBlock(num_heads, dmodel, dk, dv, dff, dropout));
        
    def forward(self, bos_X, enc_output):
        
        # X.shape = (batch_size, number of steps in bos_X, dmodel)
        X = self.pencoding(self.embedding(bos_X));
        
        mask = True if self.training else False;
        
        for i in range(self.num_blocks):
            X = self.decoder_blocks[i](X, enc_output, mask);
            
        return self.W_out(X);

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__();
        self.encoder = encoder;
        self.decoder = decoder;
        
    def forward(self, src_X, bos_X, source_seq_len):
        # src_X.shape = (batch_size, number of steps in src_X)
        # bos_X.shape = (batch_size, number of steps in bos_X)
        
        enc_output = self.encoder(src_X, source_seq_len);
        Y_hat = self.decoder(bos_X, enc_output);
        
        return Y_hat;
    
# INSTANTIATION OF MODEL AND HYPERPARAMETERS

## -- ENGLISH -> FRENCH IMPLEMENTATION -- ##
def loadVocab():
    with open("./saved_objects/vocabs_en_to_fr.pkl", "rb") as f:
        en_source_vocab = pickle.load(f);
        fr_target_vocab = pickle.load(f);

    return en_source_vocab, fr_target_vocab;

def loadModel(load_parameters=False, load_on_cpu=True):

    source_vocab, target_vocab = loadVocab();

    source_vocab_size = len(source_vocab);
    target_vocab_size = len(target_vocab);

    # SHARED HYPERPARAMETERS
    num_blocks = 2;
    num_heads = 6;
    dmodel, dk, dv, dff = 128, 32, 32, 1024;
    dropout = 0.1;

    encoder = Encoder(num_blocks, source_vocab_size, num_heads, dmodel, dk, dv, dff, dropout);
    decoder = Decoder(num_blocks, target_vocab_size, num_heads, dmodel, dk, dv, dff, dropout);
    model = EncoderDecoder(encoder, decoder);

    if load_parameters:
        if load_on_cpu:
            checkpoint = torch.load('./saved_objects/parameters_Transformer_en_to_fr.tar', map_location=torch.device('cpu'));
        else:
            checkpoint = torch.load('./saved_objects/parameters_Transformer_en_to_fr.tar');

        model.load_state_dict(checkpoint['model_state_dict']);

    if load_on_cpu == False:
        model.to(torch.device('cuda'));

    return model, source_vocab, target_vocab;


## -- ENGLISH <-> FRENCH IMPLEMENTATION -- ##
#
# def loadVocab(en_to_fr):
#     if en_to_fr:
#         with open("./saved_objects/vocabs_en_to_fr.pkl", 'rb') as f:
#             en_source_vocab = pickle.load(f);
#             fr_target_vocab = pickle.load(f);

#         return en_source_vocab, fr_target_vocab;

#     else:
#         with open("./saved_objects/vocabs_fr_to_en.pkl", 'rb') as f:
#             fr_source_vocab = pickle.load(f);
#             en_target_vocab = pickle.load(f);

#         return fr_source_vocab, en_target_vocab;

# def loadModel(en_to_fr, load_parameters=False, load_on_cpu=True):

#     source_vocab, target_vocab = loadVocab(en_to_fr);

#     source_vocab_size = len(source_vocab);
#     target_vocab_size = len(target_vocab);

#     # SHARED HYPERPARAMETERS
#     num_blocks = 2;
#     num_heads = 6;
#     dmodel, dk, dv, dff = 128, 32, 32, 512;
#     dropout = 0.1;

#     encoder = Encoder(num_blocks, source_vocab_size, num_heads, dmodel, dk, dv, dff, dropout);
#     decoder = Decoder(num_blocks, target_vocab_size, num_heads, dmodel, dk, dv, dff, dropout);
#     model = EncoderDecoder(encoder, decoder);

#     if load_parameters:
#         if load_on_cpu:
#             if en_to_fr:
#                 checkpoint = torch.load('./saved_objects/parameters_Transformer_en_to_fr.tar', map_location=torch.device('cpu'));
#             else:
#                 checkpoint = torch.load('./saved_objects/parameters_Transformer_fr_to_en.tar', map_location=torch.device('cpu'));
#         else:
#             if en_to_fr:
#                 checkpoint = torch.load('./saved_objects/parameters_Transformer_en_to_fr.tar');
#             else:
#                 checkpoint = torch.load('./saved_objects/parameters_Transformer_fr_to_en.tar');

#         model.load_state_dict(checkpoint['model_state_dict']);

#     if load_on_cpu == False:
#         model.to(torch.device('cuda'));

#     return model, source_vocab, target_vocab;
#
## ----------------------------------------------##


# PREDICTION GIVEN USER INPUT

def prediction(model,datasets,source_vocab,target_vocab):
    
    bos_idx = target_vocab.token_to_idx['<bos>'];
    eos_idx = target_vocab.token_to_idx['<eos>'];

    preds_outputs_src = [];
    preds_outputs_y = [];
    
    src_X, source_seq_len_test, Y = next(iter(datasets));    
    bos_X = torch.empty((len(src_X),1)).fill_(bos_idx).type(torch.int32);
    
    start = time.time();
    while(len(src_X) > 0):

        Y_hat = torch.transpose(model(src_X, bos_X, source_seq_len_test),0,1)[-1];
        preds = torch.argmax(Y_hat,dim=-1,keepdim=True);

        bos_X = torch.cat((bos_X,preds),dim=-1);

        ## Halt prediction if <eos> token.
        preds_is_eos = (preds == eos_idx).flatten();

        src_X_halt = source_vocab.idxToToken(src_X[preds_is_eos]);
        for i in range(len(src_X_halt)):
            preds_outputs_src.append(src_X_halt[i]);

        bos_X_halt = target_vocab.idxToToken(bos_X[preds_is_eos]);
        for i in range(len(bos_X_halt)):
            preds_outputs_y.append(bos_X_halt[i]);

        ## Delete terminated predictions.
        src_X = src_X[~preds_is_eos];
        bos_X = bos_X[~preds_is_eos];
        source_seq_len_test = source_seq_len_test[~preds_is_eos];

        if (time.time() - start) > 15:
            preds_outputs_y = None;
            break; 
         
    return preds_outputs_src, preds_outputs_y;

def standardizeOutput(output):
    if output is None:
        return "EXECUTION ERROR ON THE SERVER";
    else:
        output = output[0];

        standardized_ouput = "";
        output = output[1:-1];

        len_output = len(output);
        for i in range(len_output):
            if output[i] == '<special_begin>' or output[i] == '<special_end>':
                continue;
            elif output[i] == '<space>':
                standardized_ouput += " ";
            else:
                standardized_ouput += output[i];

        return standardized_ouput;

def translateUserInput(user_input, model, source_vocab, target_vocab):

    user_input_standardized = [standardizeString(user_input, False)];
    user_input_standardized_tokenized = source_vocab.tokenToIdx(user_input_standardized);

    user_input_sequence_len = sequencesLen(user_input_standardized);

    # torch.tensor([[0]]) is just here to replace the Y (i.e. the traduction of the user_input) that I don't have access to in production.
    data = dataLoader(1, False, user_input_standardized_tokenized, user_input_sequence_len, torch.tensor([[0]]));

    _, out_y = prediction(model, data, source_vocab, target_vocab);

    return standardizeOutput(out_y);


