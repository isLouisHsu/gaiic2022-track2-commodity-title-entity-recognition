import re
import jieba
import logging
from typing import *
from transformers.models.bert.tokenization_bert import (
    BasicTokenizer, 
    BertTokenizer, 
    WordpieceTokenizer,
    whitespace_tokenize,
)
from transformers.tokenization_utils import (
    _is_control, 
    _is_punctuation, 
    _is_whitespace,
)
from transformers.file_utils import PaddingStrategy, TensorType, add_end_docstrings
from transformers.tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    INIT_TOKENIZER_DOCSTRING,
    AddedToken,
    BatchEncoding,
    EncodedInput,
    EncodedInputPair,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)

logger = logging.getLogger(__name__)

class BasicTokenizerZh(BasicTokenizer):
    """ 中文预训练模型分词器基于BERT，会将句子中出现的空白符删除 """
    
    def __init__(
        self, 
        space_token="[unused1]", 
        do_lower_case=True, 
        do_ref_tokenize=False,
        never_split=None, 
        tokenize_chinese_chars=True, 
        strip_accents=None,
    ):
        super().__init__(
            do_lower_case=do_lower_case, 
            never_split=never_split, 
            tokenize_chinese_chars=tokenize_chinese_chars, 
            strip_accents=strip_accents
        )
        self.space_token = space_token
        self.do_ref_tokenize = do_ref_tokenize

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or \
                _is_control(char) or _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        if not self.do_ref_tokenize:
            output = []
            for char in text:
                if char == " ":
                    output.append(" ")
                    output.append(self.space_token)
                    output.append(" ")
                    continue
                cp = ord(char)
                if self._is_chinese_char(cp):
                    output.append(" ")
                    output.append(char)
                    output.append(" ")
                else:
                    output.append(char)
        else:
            output = []
            for word in self._cut_words(text):
                if word == " ":
                    output.append(" ")
                    output.append(self.space_token)
                    output.append(" ")
                    continue
                output.append(" ")
                output.append(word)
                output.append(" ")

        return "".join(output)

    def _cut_words(self, text):
        return jieba.cut(text)  # TODO: jieba, ltp, hanlp, ...

class WordpieceTokenizerZh(WordpieceTokenizer):

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.

        Returns:
            A list of wordpiece tokens.
        """

        output_tokens = []
        offsets_mapping = []
        offset_start = 0
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                # output_tokens.append(self.unk_token)
                # offset_end = offset_start + len(token)
                # offsets_mapping.append([offset_start, offset_end])
                # offset_start = offset_end
                for ch in token:
                    output_tokens.append(self.unk_token)
                    offset_end = offset_start + len(ch)
                    offsets_mapping.append([offset_start, offset_end])
                    offset_start = offset_end
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    if sub_token.startswith("##"):
                        sub_token = sub_token[2:]
                    offset_end = offset_start + len(sub_token)
                    offsets_mapping.append([offset_start, offset_end])
                    offset_start = offset_end
        return output_tokens, offsets_mapping

class BertTokenizerZh(BertTokenizer):

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_ref_tokenize=False,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        space_token="[unused1]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        self.space_token = space_token
        if never_split is None:
            never_split = [space_token,]
        else:
            never_split.append(space_token)
        
        if "do_basic_tokenize" in kwargs:
            kwargs.pop("do_basic_tokenize")
        if "additional_special_tokens" in kwargs:
            kwargs.pop("additional_special_tokens")

        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=True,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            additional_special_tokens=[space_token,],
            **kwargs
        )

        self.basic_tokenizer = BasicTokenizerZh(
            do_lower_case=do_lower_case,
            do_ref_tokenize=do_ref_tokenize,
            never_split=never_split,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
        )
        self.wordpiece_tokenizer = WordpieceTokenizerZh(
            vocab=self.vocab, 
            unk_token=self.unk_token
        )

    def _tokenize(self, text):
        split_tokens = []
        offsets_mapping = []
        offset_start = 0
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    offset_end = offset_start + 1
                    split_tokens.append(token)
                    offsets_mapping.append([offset_start, offset_end])
                    offset_start = offset_end
                else:
                    sub_split_tokens, sub_offsets_mapping = \
                        self.wordpiece_tokenizer.tokenize(token)
                    for sub_split_token, (sub_start, sub_end) in zip(
                            sub_split_tokens, sub_offsets_mapping):
                        split_tokens.append(sub_split_token)
                        sub_start, sub_end = sub_start + offset_start, sub_end + offset_start
                        offsets_mapping.append([sub_start, sub_end])
                    offset_start = sub_end
        else:
            # split_tokens = self.wordpiece_tokenizer.tokenize(text)
            raise NotImplementedError
        return split_tokens, offsets_mapping

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        is_pre_tokenized: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:

        no_split_token = set(self.unique_no_split_tokens)
        def get_input_ids(text):
            if isinstance(text, str):
                tokens, offsets_mapping = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens), offsets_mapping
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = []
                    offsets_mapping = []
                    offset_start = 0
                    for t in text:
                        if t in no_split_token:
                            tokens.append(t)
                            offset_end = offset_start + 1
                            offsets_mapping.append([offset_start, offset_end])
                            offset_start = offset_end
                        else:
                            sub_tokens, sub_offsets_mapping = self.tokenize(
                                t, is_split_into_words=True, **kwargs)
                            for sub_token, (sub_start, sub_end) in zip(
                                    sub_tokens, sub_offsets_mapping):
                                tokens.append(sub_token)
                                sub_start, sub_end = sub_start + offset_start, sub_end + offset_start
                                offsets_mapping.append([sub_start, sub_end])
                                offset_start = sub_end
                    return self.convert_tokens_to_ids(tokens), offsets_mapping
                elif is_pre_tokenized:
                    return self.convert_tokens_to_ids(text), None
                else:
                    return self.convert_tokens_to_ids(text), None
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text, None
            else:
                if is_split_into_words:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string or a list/tuple of strings when `is_split_into_words=True`."
                    )
                else:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                    )

        first_ids, first_offsets_mapping = get_input_ids(text)
        second_ids, second_offsets_mapping = get_input_ids(text_pair) if text_pair is not None else None, None

        batch_encoding = self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            offsets_mapping=first_offsets_mapping,
            pair_offsets_mapping=second_offsets_mapping,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_offsets_mapping=return_offsets_mapping,
            verbose=verbose,
        )

        return batch_encoding

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        is_pre_tokenized: bool = False, # TODO:
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:

        no_split_token = set(self.unique_no_split_tokens)
        def get_input_ids(text):
            if isinstance(text, str):
                tokens, offsets_mapping = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens), offsets_mapping
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = []
                    offsets_mapping = []
                    offset_start = 0
                    for t in text:
                        if t in no_split_token:
                            tokens.append(t)
                            offset_end = offset_start + 1
                            offsets_mapping.append([offset_start, offset_end])
                            offset_start = offset_end
                        else:
                            sub_tokens, sub_offsets_mapping = self.tokenize(
                                t, is_split_into_words=True, **kwargs)
                            for sub_token, (sub_start, sub_end) in zip(
                                    sub_tokens, sub_offsets_mapping):
                                tokens.append(sub_token)
                                sub_start, sub_end = sub_start + offset_start, sub_end + offset_start
                                offsets_mapping.append([sub_start, sub_end])
                                offset_start = sub_end
                    return self.convert_tokens_to_ids(tokens), offsets_mapping
                elif is_pre_tokenized:
                    return self.convert_tokens_to_ids(text), None
                else:
                    return self.convert_tokens_to_ids(text), None
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text, None
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        batch_ids_pairs = []
        batch_offsets_mapping_pairs = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids, first_offsets_mapping = get_input_ids(ids)
            second_ids, second_offsets_mapping = get_input_ids(pair_ids) if pair_ids is not None else None, None
            batch_ids_pairs.append((first_ids, second_ids))
            batch_offsets_mapping_pairs.append((first_offsets_mapping, second_offsets_mapping))

        batch_outputs = self._batch_prepare_for_model(
            batch_ids_pairs,
            batch_offsets_mapping_pairs=batch_offsets_mapping_pairs,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return BatchEncoding(batch_outputs)

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def _batch_prepare_for_model(
        self,
        batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
        batch_offsets_mapping_pairs: List = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        batch_outputs = {}
        for (first_ids, second_ids), (first_offset_mapping, second_offset_mapping) in \
                zip(batch_ids_pairs, batch_offsets_mapping_pairs):
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                offsets_mapping=first_offset_mapping,
                pair_offsets_mapping=second_offset_mapping,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        offsets_mapping: List[List] = None,
        pair_offsets_mapping: List[List] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens. Please Note, for *pair_ids*
        different than `None` and *truncation_strategy = longest_first* or `True`, it is not possible to return
        overflowing tokens. Such a combination of arguments will raise an error.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        if (
            return_overflowing_tokens
            and truncation_strategy == TruncationStrategy.LONGEST_FIRST
            and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )
            if return_offsets_mapping:
                raise NotImplementedError   # TODO: 截断offsets_mapping

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)
        if return_offsets_mapping:
            encoded_inputs["offset_mapping"] = self.get_offsets_mapping(offsets_mapping, pair_offsets_mapping)

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs
    
    def get_offsets_mapping(self, offsets_mapping, pair_offsets_mapping):
        if pair_offsets_mapping is None:
            offsets_mapping = [[0, 0]] + offsets_mapping + [[0, 0]]
        else:
            offsets_mapping = [[0, 0]] + offsets_mapping + [[0, 0]] + pair_offsets_mapping + [[0, 0]]
        return offsets_mapping

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        is_pre_tokenized: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
        """
        # Input type checking for clearer error
        def _is_valid_text_input(t):
            if isinstance(t, str):
                # Strings are fine
                return True
            elif isinstance(t, (list, tuple)):
                # List are fine as long as they are...
                if len(t) == 0:
                    # ... empty
                    return True
                elif isinstance(t[0], str):
                    # ... list of strings
                    return True
                elif isinstance(t[0], (list, tuple)):
                    # ... list with an empty list or with a list of strings
                    return len(t[0]) == 0 or isinstance(t[0][0], str)
                else:
                    return False
            else:
                return False

        if not _is_valid_text_input(text):
            raise ValueError(
                "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples)."
            )

        if text_pair is not None and not _is_valid_text_input(text_pair):
            raise ValueError(
                "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples)."
            )

        if is_split_into_words or is_pre_tokenized:
            is_batched = isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
        else:
            is_batched = isinstance(text, (list, tuple))

        if is_batched:
            if isinstance(text_pair, str):
                raise TypeError(
                    "when tokenizing batches of text, `text_pair` must be a list or tuple with the same length as `text`."
                )
            if text_pair is not None and len(text) != len(text_pair):
                raise ValueError(
                    f"batch length of `text`: {len(text)} does not match batch length of `text_pair`: {len(text_pair)}."
                )
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            return self.batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                is_pre_tokenized=is_pre_tokenized,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
            )
        else:
            return self.encode_plus(
                text=text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                is_pre_tokenized=is_pre_tokenized,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
            )

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # out_string = " ".join(tokens).replace(" ##", "").strip()
        out_string = "".join(tokens)
        out_string = re.sub(re.escape(self.space_token), r" ", out_string)
        out_string = re.sub(re.escape(self.unk_token), r" ", out_string)
        out_string = re.sub(r"##", r"", out_string)
        return out_string
