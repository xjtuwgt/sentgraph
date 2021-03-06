import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from transformers import LongformerTokenizer
from scripts.lb_Longformer_hotpotQAUtils import mask_generation

class HotpotTestDataset(Dataset): ##for dev dataloader
    ##ques_encode', 'ques_len', 'ctx_encode', 'ctx_lens', 'ctx_max_len'
    def __init__(self, data_frame: DataFrame, tokenizer: LongformerTokenizer, max_token_num=4096, max_doc_num=10,
                 max_sent_num=150):
        self.len = data_frame.shape[0]
        self.data = data_frame
        self.max_token_num = max_token_num
        self.max_doc_num = max_doc_num
        self.max_sent_num = max_sent_num
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        example = self.data.iloc[idx]
        example_id = torch.LongTensor([example['e_id']])  ### for ddp validation alignment
        ####
        doc_num = example['doc_num']
        concat_sent_num = example['sent_num']
        concat_len = example['ctx_len']
        ####
        cat_doc_encodes = example['ctx_encode']
        cat_doc_attention_mask = [1] * concat_len
        cat_doc_global_attn_mask = example['global_attn']
        ctx_marker_mask = example['ans_mask']

        ctx_token2sent_map = example['token2sent']
        assert concat_len == len(cat_doc_encodes)
        if concat_len < self.max_token_num:
            token_pad_num = self.max_token_num - concat_len
            cat_doc_encodes = cat_doc_encodes + [self.pad_token_id] * token_pad_num
            cat_doc_global_attn_mask = cat_doc_global_attn_mask + [0] * token_pad_num
            ctx_marker_mask = ctx_marker_mask + [0] * token_pad_num
            cat_doc_attention_mask = cat_doc_attention_mask + [0] * token_pad_num
            ctx_token2sent_map = ctx_token2sent_map + [0] * token_pad_num
        cat_doc_encodes = torch.LongTensor(cat_doc_encodes)
        cat_doc_attention_mask = torch.LongTensor(cat_doc_attention_mask)
        ctx_token2sent_map = torch.LongTensor(ctx_token2sent_map)
        cat_doc_global_attn_mask = torch.LongTensor(cat_doc_global_attn_mask)
        ctx_marker_mask = torch.BoolTensor(ctx_marker_mask)
        ################################################################################################################
        doc_start_idxes = example['doc_start']
        doc_end_idxes = example['doc_end']
        doc_lens = example['doc_len']
        doc_sent_nums = example['doc_sent_num']
        if doc_num < self.max_doc_num:
            doc_pad_num = self.max_doc_num - doc_num
            doc_start_idxes = doc_start_idxes + [0] * doc_pad_num
            doc_end_idxes = doc_end_idxes + [0] * doc_pad_num
            doc_lens = doc_lens + [0] * doc_pad_num
            doc_sent_nums = doc_sent_nums + [0] * doc_pad_num
        doc_start_idxes = torch.LongTensor(doc_start_idxes)
        doc_end_idxes = torch.LongTensor(doc_end_idxes)
        doc_lens = torch.LongTensor(doc_lens)
        ################################################################################################################
        sent_start_idxes = example['sent_start']
        sent_end_idxes = example['sent_end']
        ctx_sent_lens = example['sent_len']
        ctx_sent2doc_map = example['sent2doc']
        ctx_sentIndoc_idx = example['sentIndoc']
        if concat_sent_num < self.max_sent_num:
            sent_pad_num = self.max_sent_num - concat_sent_num
            sent_start_idxes = sent_start_idxes + [0] * sent_pad_num
            sent_end_idxes = sent_end_idxes + [0] * sent_pad_num
            ctx_sent_lens = ctx_sent_lens + [0] * sent_pad_num
            ctx_sent2doc_map = ctx_sent2doc_map + [0] * sent_pad_num
            ctx_sentIndoc_idx = ctx_sentIndoc_idx + [0] * sent_pad_num
        sent_start_idxes = torch.LongTensor(sent_start_idxes)
        sent_end_idxes = torch.LongTensor(sent_end_idxes)
        ctx_sent_lens = torch.LongTensor(ctx_sent_lens)
        ctx_sent2doc_map = torch.LongTensor(ctx_sent2doc_map)
        ctx_sentIndoc_idx = torch.LongTensor(ctx_sentIndoc_idx)
        ################################################################################################################
        ss_attn_mask, sd_attn_mask = mask_generation(sent_num_docs=doc_sent_nums, max_sent_num=self.max_sent_num)
        ################################################################################################################
        ################################################################################################################
        assert concat_len <= self.max_token_num
        ################################################################################################################
        return cat_doc_encodes, cat_doc_attention_mask, cat_doc_global_attn_mask, doc_start_idxes, sent_start_idxes, \
               doc_lens, ctx_sent_lens, ss_attn_mask, sd_attn_mask, ctx_sent2doc_map, ctx_sentIndoc_idx, ctx_token2sent_map, ctx_marker_mask, \
               doc_end_idxes, sent_end_idxes, concat_len, concat_sent_num, example_id

    @staticmethod
    def collate_fn(data):
        batch_max_ctx_len = max([_[15] for _ in data])
        batch_max_sent_num = max([_[16] for _ in data])
        batch_ids = torch.stack([_[17] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ctx_sample = torch.stack([_[0] for _ in data], dim=0)
        batch_ctx_mask_sample = torch.stack([_[1] for _ in data], dim=0)
        batch_ctx_global_sample = torch.stack([_[2] for _ in data], dim=0)

        batch_ctx_sample = batch_ctx_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_mask_sample = batch_ctx_mask_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_global_sample = batch_ctx_global_sample[:, range(0, batch_max_ctx_len)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_starts = torch.stack([_[3] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent_starts = torch.stack([_[4] for _ in data], dim=0)
        batch_sent_starts = batch_sent_starts[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_lens = torch.stack([_[5] for _ in data], dim=0)
        batch_sent_lens = torch.stack([_[6] for _ in data], dim=0)
        batch_sent_lens = batch_sent_lens[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ss_attn_mask = torch.stack([_[7] for _ in data], dim=0)
        batch_sd_attn_mask = torch.stack([_[8] for _ in data], dim=0)
        batch_ss_attn_mask = batch_ss_attn_mask[:, range(0, batch_max_sent_num)][:, :, range(0, batch_max_sent_num)]
        batch_sd_attn_mask = batch_sd_attn_mask[:, :, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent2doc_map = torch.stack([_[9] for _ in data], dim=0)
        batch_sent2doc_map = batch_sent2doc_map[:, range(0, batch_max_sent_num)]
        batch_sentIndoc_map = torch.stack([_[10] for _ in data], dim=0)
        batch_sentIndoc_map = batch_sentIndoc_map[:, range(0, batch_max_sent_num)]
        batch_token2sent = torch.stack([_[11] for _ in data], dim=0)
        batch_token2sent = batch_token2sent[:, range(0, batch_max_ctx_len)]
        batch_marker = torch.stack([_[12] for _ in data], dim=0)
        batch_marker = batch_marker[:, range(0, batch_max_ctx_len)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_ends = torch.stack([_[13] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent_ends = torch.stack([_[14] for _ in data], dim=0)
        batch_sent_ends = batch_sent_ends[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        res = {'ctx_encode': batch_ctx_sample, 'ctx_attn_mask': batch_ctx_mask_sample,
               'ctx_global_mask': batch_ctx_global_sample, 'doc_start': batch_doc_starts, 'doc_end': batch_doc_ends,
               'sent_start': batch_sent_starts, 'sent_end': batch_sent_ends, 'id': batch_ids,
               'doc_lens': batch_doc_lens, 'sent_lens': batch_sent_lens, 'ss_mask': batch_ss_attn_mask, 'sd_mask': batch_sd_attn_mask,
               's2d_map': batch_sent2doc_map, 'sInd_map': batch_sentIndoc_map, 'marker': batch_marker,
               't2s_map': batch_token2sent}
        return res