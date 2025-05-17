import torch
import torch.nn as nn
import torch.nn.functional as F
from . import transformer as tr

from .models import META_ARCHITECTURES as registry
from .normalized_linear import NormalizedLinear
from .feature_head import BaseFeatureHead_mat
from ..utils.ek_utils import (action_to_noun_map, action_to_verb_map)


@registry.register('CMeRT')
class CMeRT(nn.Module):
    # contextual enhancement
    def __init__(self, cfg):
        super(CMeRT, self).__init__()

        self.cfg = cfg
        # Build long feature heads
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.long_enabled = self.long_memory_num_samples > 0
        if self.long_enabled:
            self.feature_head_long = BaseFeatureHead_mat(cfg)

        # Build work feature head
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.work_enabled = self.work_memory_num_samples > 0
        if self.work_enabled:
            self.feature_head_work = BaseFeatureHead_mat(cfg)

        self.d_model = self.feature_head_work.d_model
        self.num_heads = cfg.MODEL.LSTR.NUM_HEADS
        self.dim_feedforward = cfg.MODEL.LSTR.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.LSTR.DROPOUT
        self.activation = cfg.MODEL.LSTR.ACTIVATION
        self.num_classes = cfg.DATA.NUM_CLASSES

        # Build position encoding
        self.pos_encoding = tr.PositionalEncoding(self.d_model, self.dropout)

        self.anticipation_length = cfg.MODEL.LSTR.ANTICIPATION_LENGTH
        self.anticipation_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.anticipation_num_samples = self.anticipation_length // self.anticipation_sample_rate
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES
        self.future_enabled = self.future_num_samples > 0
        self.context_num_samples = int(cfg.MODEL.LSTR.CONTEXT_SECONDS * cfg.DATA.FPS) // cfg.MODEL.LSTR.CONTEXT_MEMORY_SAMPLE_RATE

        self.long_memory_use_pe = cfg.MODEL.LSTR.LONG_MEMORY_USE_PE
        self.work_memory_use_pe = cfg.MODEL.LSTR.WORK_MEMORY_USE_PE
        self.include_work = cfg.MODEL.LSTR.LONG_MEMORY_INCLUDE_WORK

        # Build LSTR encoder
        if self.long_enabled:
            self.enc_queries = nn.ModuleList()
            self.enc_modules = nn.ModuleList()
            for i, param in enumerate(cfg.MODEL.LSTR.ENC_MODULE):
                if param[0] != -1:
                    self.enc_queries.append(nn.Embedding(param[0], self.d_model))
                    enc_layer = tr.TransformerDecoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation)
                    self.enc_modules.append(tr.TransformerDecoder(
                        enc_layer, param[1], tr.layer_norm(self.d_model, param[2])))
                else:
                    self.enc_queries.append(None)
                    enc_layer = tr.TransformerEncoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation)  # ,attention_type=self.encoder_attention_type
                    self.enc_modules.append(tr.TransformerEncoder(
                        enc_layer, param[1], tr.layer_norm(self.d_model, param[2])))
        else:
            self.register_parameter('enc_queries', None)
            self.register_parameter('enc_modules', None)

        # Build LSTR decoder
        param = cfg.MODEL.LSTR.DEC_MODULE
        dec_layer = tr.TransformerDecoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation)
        self.dec_modules = tr.TransformerDecoder(
                dec_layer, param[1], tr.layer_norm(self.d_model, param[2]))

        # Build Future Generation
        if self.future_enabled:
            param = cfg.MODEL.LSTR.GEN_MODULE
            self.gen_query = nn.Embedding(param[0], self.d_model)
            gen_layer = tr.TransformerDecoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation)
            self.gen_layer = tr.TransformerDecoder(
                gen_layer, param[1], tr.layer_norm(self.d_model, param[2])
            )
            self.final_query = nn.Embedding(cfg.MODEL.LSTR.FUT_MODULE[0][0], self.d_model)

            #
            self.work_fusions = nn.ModuleList()
            for i in range(cfg.MODEL.LSTR.REFINE_TIMES):
                work_enc_layer = tr.TransformerDecoderLayer(
                    self.d_model, self.num_heads, self.dim_feedforward,
                    self.dropout, self.activation)
                self.work_fusions.append(tr.TransformerDecoder(
                    work_enc_layer, 1, tr.layer_norm(self.d_model, True)))

        else:
            if self.anticipation_num_samples > 0 :
                self.final_query = nn.Embedding(self.anticipation_num_samples*3, self.d_model)
        # Build classifier
        if cfg.MODEL.LSTR.DROPOUT_CLS > 0:
            self.dropout_cls = nn.Dropout(cfg.MODEL.LSTR.DROPOUT_CLS)
        else:
            self.dropout_cls = None
        if cfg.MODEL.LSTR.FC_NORM:
            self.classifier = NormalizedLinear(self.d_model, self.num_classes)
        else:
            self.classifier = nn.Linear(self.d_model, self.num_classes)

    def refine(self, memory, output, mask):
        his_memory = memory
        enc_query = self.gen_query.weight.unsqueeze(1).repeat(1, his_memory.shape[1], 1)
        future = self.gen_layer(enc_query, his_memory)  # , knn=True

        future_rep = [future]
        short_rep = [output]
        for i in range(self.cfg.MODEL.LSTR.REFINE_TIMES):
            mask1 = torch.zeros((output.shape[0], memory.shape[0])).to(output.device)
            mask2 = torch.zeros((output.shape[0], future.shape[0])).to(output.device)
            the_mask = torch.cat((mask1, mask, mask2), dim=-1)
            total_memory = torch.cat([memory, output, future])
            output = self.work_fusions[i](output, total_memory, tgt_mask=mask, memory_mask=the_mask)  # , knn=True
            short_rep.append(output)

        return short_rep, future_rep

    def forward(self, visual_inputs, motion_inputs, memory_key_padding_mask=None):
        if self.long_enabled:
            # Compute long memories
            if self.long_memory_use_pe:
                long_memories = self.pos_encoding(self.feature_head_long(
                    visual_inputs[:, :self.long_memory_num_samples],
                    motion_inputs[:, :self.long_memory_num_samples]
                ).transpose(0, 1))
                long_memories = long_memories.transpose(0, 1)  ## transpose back
            else:
                long_memories = self.feature_head_long(
                    visual_inputs[:, :self.long_memory_num_samples],
                    motion_inputs[:, :self.long_memory_num_samples]
                )
            batch_size = long_memories.shape[0]
            long_memories = long_memories.view(batch_size, -1, self.d_model).transpose(0, 1)

            if len(self.enc_modules) > 0:
                enc_queries = [
                    enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                    if enc_query is not None else None
                    for enc_query in self.enc_queries
                ]

                # Encode long memories
                if enc_queries[0] is not None:
                    long_memories = self.enc_modules[0](enc_queries[0], long_memories,
                                                        memory_key_padding_mask=memory_key_padding_mask)
                else:
                    long_memories = self.enc_modules[0](long_memories)
                for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                    if enc_query is not None:
                        long_memories = enc_module(enc_query, long_memories)
                    else:
                        long_memories = enc_module(long_memories)

        # Concatenate memories
        if self.long_enabled:
            memory = long_memories

        # Compute work memories
        visual_inputs_avg = visual_inputs[:, self.long_memory_num_samples:]
        motion_inputs_avg = motion_inputs[:, self.long_memory_num_samples:]
        work_memories_no_pe = self.feature_head_work(
            visual_inputs_avg,
            motion_inputs_avg,
        ).transpose(0, 1)
        if self.work_memory_use_pe:
            work_memories = self.pos_encoding(work_memories_no_pe,
                                              padding=self.long_memory_num_samples if self.long_memory_use_pe else 0)
        else:
            work_memories = work_memories_no_pe

        if self.anticipation_num_samples > 0:
            anticipate_memories = self.pos_encoding(
                self.final_query.weight[
                :self.cfg.MODEL.LSTR.ANTICIPATION_LENGTH:self.cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE, ...]. \
                    unsqueeze(1).repeat(1, work_memories.shape[1], 1),
                padding=self.long_memory_num_samples + self.work_memory_num_samples if self.long_memory_use_pe else self.work_memory_num_samples)
            work_memories = torch.cat((work_memories, anticipate_memories), dim=0)

        # Build mask
        mask = tr.generate_square_subsequent_mask(work_memories.shape[0]).to(work_memories.device)
        if self.anticipation_num_samples > 0:
            mask1 = torch.zeros((self.anticipation_num_samples, self.anticipation_num_samples)).to(work_memories.device)
            mask[-self.anticipation_num_samples:, -self.anticipation_num_samples:] = mask1
        # Compute output
        if self.long_enabled:
            if self.include_work:
                extended_memory = torch.cat((memory, work_memories), dim=0)
                extended_mask = F.pad(mask,
                                      (memory.shape[0], 0),
                                      'constant', 0)
                extended_mask = extended_mask.to(extended_memory.device)
                output = self.dec_modules(
                    work_memories,
                    memory=extended_memory,
                    tgt_mask=mask,
                    memory_mask=extended_mask,
                )
            else:
                output = self.dec_modules(
                    work_memories,
                    memory=memory,
                    tgt_mask=mask,
                )
        else:
            output = self.dec_modules(
                work_memories,
                src_mask=mask,
            )

        if self.future_enabled:
            works, futs = self.refine(memory, output[self.context_num_samples:], mask[self.context_num_samples:, self.context_num_samples:])
            work_scores = []
            fut_scores = []
            for i, work in enumerate(works):
                work_scores.append(self.classifier(work).transpose(0, 1))
            for i, fut in enumerate(futs):
                if i == 0:
                    fut_scores.append(self.classifier(
                        F.interpolate(fut.permute(1, 2, 0), size=self.future_num_samples).permute(2, 0, 1)).transpose(0,
                                                                                                                      1))
                else:
                    fut_scores.append(self.classifier(fut).transpose(0, 1))
            return (work_scores, fut_scores)

        else:
            work_scores = []
            fut_scores = []
            work_scores.append(self.classifier(output[self.context_num_samples:]).transpose(0, 1))
            return work_scores, fut_scores
