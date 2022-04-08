import re
import logging
from typing import *
from transformers.models.bert.tokenization_bert import (
    BasicTokenizer, 
    BertTokenizer, 
    whitespace_tokenize,
)
from transformers.tokenization_utils import (
    _is_control, 
    _is_punctuation, 
    _is_whitespace,
)


logger = logging.getLogger(__name__)

class BasicTokenizerZh(BasicTokenizer):
    """ 中文预训练模型分词器基于BERT，会将句子中出现的空白符删除 """
    
    def __init__(self, space_token="[unused1]", do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
        super().__init__(do_lower_case=do_lower_case, never_split=never_split, tokenize_chinese_chars=tokenize_chinese_chars, strip_accents=strip_accents)
        self.space_token = space_token

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
        text = re.sub(r" ", self.space_token, text)
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        text = re.sub(re.escape(self.space_token), " " + self.space_token + " ", text)
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

class BertTokenizerZh(BertTokenizer):

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
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
            never_split=never_split,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
        )

