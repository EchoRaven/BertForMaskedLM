import math
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
import json


class BertConfig:
    def __init__(self,
                 vocab_size=1024,
                 hidden_size=768,
                 pad_token_id=100,
                 initializer_range=0.02,
                 max_position_embeddings=512,
                 type_vocab_size=1024,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 num_attention_heads=12,
                 layer_norm_eps=1e-5,
                 hidden_act="gelu",
                 intermediate_size=2304,
                 num_hidden_layers=12
                 ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

    def from_file(self, filename):
        with open(filename, "r") as config_file:
            config_data = json.load(config_file)
        self.vocab_size = config_data["vocab_size"]
        self.hidden_size = config_data["hidden_size"]
        self.pad_token_id = config_data["pad_token_id"]
        self.initializer_range = config_data["initializer_range"]
        self.max_position_embeddings = config_data["max_position_embeddings"]
        self.type_vocab_size = config_data["type_vocab_size"]
        self.hidden_dropout_prob = config_data["hidden_dropout_prob"]
        self.attention_probs_dropout_prob = config_data["attention_probs_dropout_prob"]
        self.num_attention_heads = config_data["num_attention_heads"]
        self.layer_norm_eps = config_data["layer_norm_eps"]
        self.hidden_act = config_data["hidden_act"]
        self.intermediate_size = config_data["intermediate_size"]
        self.num_hidden_layers = config_data["num_hidden_layers"]


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
            self,
            input_ids=None,
            position_ids=None,
            token_type_ids=None,
            inputs_embeds=None,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, 0: seq_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 如果hidden_size不是heads数量的整数倍，报错
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # x : [batch_size, length, hidden_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # new_x_shape : [batch_size, length, num_attention_heads, attention_head_size]
        x = x.view(new_x_shape)
        # output : [batch_size, num_attention_heads, length, attention_head_size]
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
    ) -> torch.Tensor:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # 注意，attention_mask的形式是加算不是乘算，这里需要配合softmax使用
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = context_layer
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None
    ) -> torch.Tensor:
        self_outputs = self.self(
            hidden_states,
            attention_mask
        )
        output = self.output(self_outputs, hidden_states)
        return output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if config.hidden_act == "gelu":
            self.intermediate_act_fn = nn.functional.gelu
        elif config.hidden_act == "relu":
            self.intermediate_act_fn = torch.nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None
    ) -> torch.Tensor:
        """
        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len]
        :return: [src_len, batch_size, hidden_size]
        """
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None
    ) -> list:
        """
        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len]
        :return:
        """
        all_encoder_layers = []
        layer_output = hidden_states
        for i, layer_module in enumerate(self.layer):
            layer_output = layer_module(layer_output,
                                        attention_mask)
            all_encoder_layers.append(layer_output)
        return all_encoder_layers


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.config = config
        self.dtype = torch.float32

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None
    ) -> torch.Tensor:
        """
        :param inputs_embeds:
        :param input_ids:  [src_len, batch_size]
        :param attention_mask: [batch_size, src_len]
        :param token_type_ids: [batch_size, src_len]
        :param position_ids: [1,src_len]
        :return:
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        all_encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask
        )
        sequence_output = all_encoder_outputs[-1]  # 取最后一层
        return sequence_output

    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor, dtype: torch.float = None
    ) -> torch.Tensor:
        if dtype is None:
            dtype = self.dtype
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if config.hidden_act == "gelu":
            self.transform_act_fn = nn.functional.gelu
        elif config.hidden_act == "relu":
            self.transform_act_fn = torch.nn.ReLU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class MyBertForMaskedLM(nn.Module):
    def __init__(self, config=BertConfig()):
        super().__init__()
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None
    ) -> torch.Tensor:
        """
        :param inputs_embeds:
        :param input_ids:  [src_len, batch_size]
        :param attention_mask: [batch_size, src_len]
        :param token_type_ids: [batch_size, src_len]
        :param position_ids: [1,src_len]
        :return:
        """
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds
        )
        cls_output = self.cls(bert_output)
        return cls_output

    def from_pretrained(self, from_path):
        pretrained_model = AutoModelForMaskedLM.from_pretrained(from_path)
        name_dict = {}
        pretrained_dict = pretrained_model.state_dict()
        for name in pretrained_dict:
            name_dict[name] = pretrained_dict[name]
        for name, param in self.named_parameters():
            param.requires_grad = False
            if name in name_dict.keys():
                param.copy_(name_dict[name])
            param.requires_grad = True
