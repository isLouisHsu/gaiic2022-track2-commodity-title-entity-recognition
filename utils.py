LABEL2MEANING_MAP = {
    # 主体商品，即上架的商品
    "1": "主体商品-品牌",
    "2": "主体商品-系列",
    "3": "主体商品-型号",
    "4": "主体商品-名称",
    "5": "主体商品-用途",
    "6": "主体商品-时间",
    "7": "主体商品-地点",
    "8": "主体商品-人群",
    "9": "主体商品-用途名词",
    "10": "主体商品-周边",
    "11": "主体商品-功能",
    "12": "主体商品-材料",
    "13": "主体商品-样式",
    "14": "主体商品-风格",
    "15": "主体商品-产地",
    "16": "主体商品-颜色",
    "17": "主体商品-味道",
    "18": "主体商品-尺寸",
    # 配件商品，即主体商品包含的子商品
    "19": "配件商品-品牌",
    "20": "配件商品-系列",
    "21": "配件商品-型号",
    "22": "配件商品-名称",
    "23": "配件商品-用途",
    "24": "配件商品-时间",
    "25": "配件商品-地点",
    "26": "配件商品-人群",
    # "27": "配件商品-用途名词",
    "28": "配件商品-周边",
    "29": "配件商品-功能",
    "30": "配件商品-材料",
    "31": "配件商品-样式",
    "32": "配件商品-风格",
    "33": "配件商品-产地",
    "34": "配件商品-颜色",
    "35": "配件商品-味道",
    "36": "配件商品-尺寸",
    # 其他商品，包括适用商品、赠送商品
    "37": "其他商品-品牌",
    "38": "其他商品-系列",
    "39": "其他商品-型号",
    "40": "其他商品-名称",
    "41": "其他商品-用途",
    "42": "其他商品-时间",
    "43": "其他商品-地点",
    "44": "其他商品-人群",
    # "45": "其他商品-用途名词",
    "46": "其他商品-周边",
    "47": "其他商品-功能",
    "48": "其他商品-材料",
    "49": "其他商品-样式",
    "50": "其他商品-风格",
    "51": "其他商品-产地",
    "52": "其他商品-颜色",
    "53": "其他商品-味道",
    "54": "其他商品-尺寸",
}

MEANING2LABEL_MAP = {v: k for k, v in LABEL2MEANING_MAP.items()}

def get_spans_bio(tags, id2label=None):
    """Gets entities from sequence.
    Args:
        tags (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> tags = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_spans_bio(tags)
        # output [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(tags):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx     # FIXED: 该函数无法提取由"B-X"标记的单个token实体
            chunk[0] = tag.split('-')[1]
            if indx == len(tags) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(tags) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks
