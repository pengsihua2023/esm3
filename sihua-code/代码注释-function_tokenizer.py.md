##  function_tokenizer.py

```
import re  # 导入正则表达式库，用于处理文本匹配
import string  # 导入字符串处理库
from functools import cache, cached_property, partial  # 导入功能库，用于缓存、延迟计算属性和部分函数应用
from typing import Collection  # 导入类型注解库

import numpy as np  # 导入NumPy库，用于数学和矩阵运算
import pandas as pd  # 导入Pandas库，用于数据处理和CSV文件读取
import scipy.sparse as sp  # 导入SciPy的稀疏矩阵模块
import torch  # 导入PyTorch库，用于深度学习和张量运算
import torch.nn.functional as F  # 导入PyTorch的函数式接口

from esm.tokenization.tokenizer_base import EsmTokenizerBase  # 导入基础分词器类
from esm.utils.constants import esm3 as C  # 导入常量配置
from esm.utils.function import interpro, lsh, tfidf  # 导入特定功能模块，如LSH哈希和TF-IDF
from esm.utils.misc import stack_variable_length_tensors  # 导入辅助函数
from esm.utils.types import FunctionAnnotation  # 导入功能注释的类型定义

class InterProQuantizedTokenizer(EsmTokenizerBase):  # 定义一个基于EsmTokenizerBase的分词器类
    # 类定义部分详细说明了分词器的功能和方法
    def __init__(self, depth, lsh_bits_per_token, lsh_path, keyword_vocabulary_path, keyword_idf_path, interpro_entry_path, interpro2keywords_path):
        # 构造函数初始化分词器的参数
        # 参数包括深度、LSH位数、文件路径等
    @cached_property
    def interpro2keywords(self):  # 延迟计算属性，从CSV文件读取InterPro ID到关键词的映射
    @cached_property
    def interpro_labels(self):  # 延迟计算属性，返回支持的InterPro标签
    @cached_property
    def interpro_to_index(self):  # 延迟计算属性，生成InterPro ID到索引的映射
    @property
    def keyword_vocabulary(self):  # 属性，返回支持的关键词
    @property
    def keyword_to_index(self):  # 属性，生成关键词到索引的映射
    @cached_property
    def _tfidf(self):  # 延迟计算属性，创建TF-IDF模型
    @cached_property
    def special_tokens(self):  # 延迟计算属性，定义特殊标记列表
    @cached_property
    def vocab(self):  # 延迟计算属性，生成完整的词汇表
    @cached_property
    def vocab_to_index(self):  # 延迟计算属性，生成词汇到索引的映射
    def get_special_tokens_mask(self, encoded):  # 返回特殊标记的掩码
    def tokenize(self, annotations, seqlen, p_keyword_dropout):  # 根据注释对蛋白质功能进行标记化
    def _function_text_hash(self, labels, keyword_mask):  # 使用LSH对功能文本进行哈希
    def encode(self, tokens, add_special_tokens):  # 将标记转换为ID张量
    def lookup_annotation_name(self, annotation):  # 查找注释的名称
    def format_annotation(self, annotation):  # 格式化注释
    def _token2ids(self, token):  # 将标记转换为ID集
    def batch_encode(self, token_batch, add_special_tokens):  # 批量编码功能标记
    def decode(self, encoded):  # 解码功能标记
    @property
    def mask_token(self):  # 定义掩码标记
    @property
    def mask_token_id(self):  # 定义掩码标记ID
    @property
    def bos_token(self):  # 定义开始标记
    @property
    def bos_token_id(self):  # 定义开始标记ID
    @property
    def eos_token(self):  # 定义结束标记
    @property
    def eos_token_id(self):  # 定义结束标记ID
    @property
    def pad_token(self):  # 定义填充标记
    @property
    def pad_token_id(self):  # 定义填充标记ID

def _texts_to_keywords(texts):  # 将文本描述转换为关键词集
def _keywords_from_text(text):  # 将文本拆分为单词和双词
def _sanitize(text):  # 清理文本，移除标点和特定单词

_EXCLUDED_TERMS = {  # 定义在文本表示中省略的常见但不具体的术语
    # 列出了多种常见的术语，如“binding domain”和“molecular function”等
    "binding domain",
    "biological_process",
    "biological process",
    "biologicalprocess",
    "c",
    "cellular_component",
    "cellular component",
    "cellularcomponent",
    "cellular_process",
    "cellularprocess",
    "cellular process",
    "cellularprocess",
    "like domain",
    "molecular function",
    "molecular_function",
    "molecularfunction",
    "n",
}


```
### 总结
这个类InterProQuantizedTokenizer继承自EsmTokenizerBase，主要用于将InterPro或功能关键词的注释转换为多标记表示形式。这通过使用TF-IDF向量的局部敏感哈希（LSH）实现。代码中定义了多个属性和方法来处理词汇、索引映射、TF-IDF编码和LSH计算，以及处理特殊标记和编码序列。此外，它还处理了文件路径和配置，确保灵活性和扩展性。  
