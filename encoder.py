import torch
import torch.nn as nn
import pickle
from ast import literal_eval

class Encoder(nn.Module):

    def __init__(self, config, text_encoder, img_encoder, attn_encoder):
        super(Encoder, self).__init__()
        self.text_encoder = text_encoder
        self.img_encoder = img_encoder
        self.attn_encoder = attn_encoder
        self.config = config

    def return_batch_from_index(self, batch):
        """
        Arguments
        ---------
        batch: Dictionary
            This accepts a batch and then returns the generated graph embeddings from the saved file based on the img_ids.
        Returns
        -------
        attn_encoder output: a tuple of the following
            im: torch.FloatTensor   
                The representation of image utility
                Shape [batch_size x NH, K, hidden_size]
            qe: torch.FloatTensor
                The representation of question utility
                Shape [batch_size x NH, N, hidden_size]
            hi: torch.FloatTensor
                The representation of history utility
                Shape [batch_size x NH, T, hidden_size]
            mask_im: torch.LongTensor
                Shape [batch_size x NH, K]
            mask_qe: torch.LongTensor
                Shape [batch_size x NH, N]
            mask_hi: torch.LongTensor
                Shape [batch_size x NH, T]
        """
        encoder_outputs = None
        # load the saved file
        # for gog
        # embeddings = pickle.load(open('batch_input_gog.pkl', 'rb'))
        
        # for master node
        # embeddings = pickle.load(open('batch_input.pkl', 'rb'))
        
        # for control
        embeddings = pickle.load(open('batch_input_control.pkl', 'rb'))

        # img_ids is the primary key for the embeddings
        batch_key = batch['img_ids'].tolist()
        # looping through the keys to find the matching img_id
        for key, val in embeddings.items():
            # since the keys are lists stored as strings, we need to convert them back to lists
            embedding_key = sorted(literal_eval(key))
            if sorted(batch_key)[0] in embedding_key:
                encoder_outputs = val
        if encoder_outputs is None:
            raise ValueError('The img_id does not match any of the keys in the embeddings file', batch_key)

        # return the embeddings where the img_id matches the batch
        # encoder_outputs = embeddings[int(batch['img_ids'][0])]
        # load encoder outputs to the batch
        batch['im'] = encoder_outputs[0]
        batch['qe'] = encoder_outputs[1]
        batch['hi'] = encoder_outputs[2]
        batch['mask_im'] = encoder_outputs[3]
        batch['mask_qe'] = encoder_outputs[4]
        batch['mask_hi'] = encoder_outputs[5]
        return batch

    def forward(self, batch, test_mode=False):
        """
        Arguments
        ---------
        batch: Dictionary
            This provides a dictionary of inputs.
        test_mode: Boolean
            Whether the forward is performed on test data
        Returns
        -------
        batch_output: a tuple of the following
            im: torch.FloatTensor
                The representation of image utility
                Shape [batch_size x NH, K, hidden_size]
            qe: torch.FloatTensor
                The representation of question utility
                Shape [batch_size x NH, N, hidden_size]
            hi: torch.FloatTensor
                The representation of history utility
                Shape [batch_size x NH, T, hidden_size]
            mask_im: torch.LongTensor
                Shape [batch_size x NH, K]
            mask_qe: torch.LongTensor
                Shape [batch_size x NH, N]
            mask_hi: torch.LongTensor
                Shape [batch_size x NH, T]

        It is noted
            K is num_proposals,
            T is the number of rounds
            N is the max sequence length in the question.
        """
        # [BS x NH, T, HS] hist
        # [BS x NH, N, HS] ques
        # [BS x NH, T] hist_mask
        # [BS x NH, N] ques_mask
        '''hist, ques, hist_mask, ques_mask = self.text_encoder(
            batch, test_mode=test_mode)'''

        # [BS x NH, K, HS] img
        # [BS x NH, K] img_mask
        '''img, img_mask = self.img_encoder(batch, test_mode=test_mode)'''

        batch_embeddings = self.return_batch_from_index(batch)
        hist = batch_embeddings['hi'].to(torch.device('cuda'))
        ques = batch_embeddings['qe'].to(torch.device('cuda'))
        img = batch_embeddings['im'].to(torch.device('cuda'))
        hist_mask = batch_embeddings['mask_hi'].to(torch.device('cuda'))
        ques_mask = batch_embeddings['mask_qe'].to(torch.device('cuda'))
        img_mask = batch_embeddings['mask_im'].to(torch.device('cuda'))

        batch_input = img, ques, hist, img_mask, ques_mask, hist_mask
        
        batch_output = self.attn_encoder(batch_input)
        return batch_output