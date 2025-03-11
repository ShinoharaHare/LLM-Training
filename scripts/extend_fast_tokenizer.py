import json

import fire
from tokenizers import Tokenizer, decoders, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def main(
    tokenizer_path: str,
    vocab_path: str,
    output_path: str,
    add_missing_vocab: bool = True,
    pad_token: str | None = None
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    with open(vocab_path) as f:
        vocab = json.load(f)

    new_tokenizer = extend_vocab(tokenizer, vocab, add_missing_vocab=add_missing_vocab)
    new_tokenizer.slow_tokenizer_class = None
    
    if pad_token is not None:
        new_tokenizer.pad_token = pad_token
    
    new_tokenizer.save_pretrained(output_path)


def get_byte_level_pre_tokenizer(tokenizer: PreTrainedTokenizerFast) -> pre_tokenizers.ByteLevel | None:
    rust_tokenizer = tokenizer.backend_tokenizer
    tokenizer_json = json.loads(rust_tokenizer.to_str())
    pre_tokenizer_json = tokenizer_json['pre_tokenizer']

    if pre_tokenizer_json is None:
        return None
    
    pre_tokenizers_ = []

    if pre_tokenizer_json['type'] == 'ByteLevel':
        pre_tokenizers_ = [pre_tokenizer_json]
    elif pre_tokenizer_json['type'] == 'Sequence':
        pre_tokenizers_ = pre_tokenizer_json['pretokenizers']

    pre_tokenizer = None
    for pt in pre_tokenizers_:
        if pt['type'] == 'ByteLevel':
            kwargs = {k: v for k, v in pt.items() if k != 'type'}
            pre_tokenizer = pre_tokenizers.ByteLevel(**kwargs)
    
    return pre_tokenizer


def get_byte_level_decoder(tokenizer: PreTrainedTokenizerFast) -> decoders.ByteLevel | None:
    rust_tokenizer = tokenizer.backend_tokenizer
    tokenizer_json = json.loads(rust_tokenizer.to_str())
    decoder_json = tokenizer_json['decoder']
    
    decoders_ = []
    if decoder_json['type'] == 'ByteLevel':
        decoders_ = [decoder_json]
    elif decoder_json['type'] == 'Sequence':
        decoders_ = decoder_json['decoders']

    decoder = None
    for d in decoders_:
        if d['type'] == 'ByteLevel':
            kwargs = {k: v for k, v in d.items() if k != 'type'}
            decoder = decoders.ByteLevel(**kwargs)
    
    return decoder


def iterate_vocab(vocab: dict[str, int]):
    for k, v in sorted(vocab.items(), key=lambda x: x[1]):
        yield k, v


def compute_merges(
    vocab: dict[str, int],
    decoder: decoders.ByteLevel | None = None
) -> tuple[list[tuple[str, str]], set[str]]:
    merges = []
    missing_vocab = set()
    for merge, piece_score in vocab.items():
        local = []
        missing_pieces = []
        for index in range(1, len(merge)):
            piece_l, piece_r = merge[:index], merge[index:]
            if piece_l in vocab and piece_r in vocab:
                local.append((piece_l, piece_r, piece_score))
            else:
                missing_pieces.append((piece_l, piece_r))
        local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]))
        missing_pieces = sorted(missing_pieces, key=lambda x: (vocab.get(x[0], len(vocab)), (vocab.get(x[1], len(vocab)))))

        if len(merges) > 1 and not local:
            merge_str = merge.replace('\n', '\\n').replace('\x9d', '\\x9d')
            print(f'Missing merge: {merge_str}', end='')

            if decoder is not None:
                print(f'({decoder.decode([merge])})', end='')
            
            print()
            
            if missing_pieces:
                piece_l, piece_r = missing_pieces[0]
                if piece_l not in vocab:
                    missing_vocab.add(piece_l)
                if piece_r not in vocab:
                    missing_vocab.add(piece_r)

        merges.extend(local)

    merges = sorted(merges, key=lambda val: val[2], reverse=True)
    merges = [(m[0], m[1]) for m in merges]
    return merges, missing_vocab


def extend_vocab(
    tokenizer: PreTrainedTokenizerFast,
    vocab: dict[str, int],
    add_missing_vocab: bool
) -> PreTrainedTokenizerFast:
    byte_level_pre_tokenizer = get_byte_level_pre_tokenizer(tokenizer)
    byte_level_decoder = get_byte_level_decoder(tokenizer)

    if byte_level_pre_tokenizer is not None:
        vocab_tmp = {}
        for k, v in iterate_vocab(vocab):
            k = byte_level_pre_tokenizer.pre_tokenize_str(k)[0][0]
            vocab_tmp[k] = len(vocab)
        vocab = vocab_tmp
    
    tokenizer_json = json.loads(tokenizer.backend_tokenizer.to_str())
    added_vocab = tokenizer.get_added_vocab()
    tokenizer_vocab = tokenizer_json['model']['vocab']

    # Make sure added tokens are in the vocab
    for k, v in iterate_vocab(added_vocab):
        if k in tokenizer_vocab:
            continue
        tokenizer_vocab[k] = v

    for k, v in iterate_vocab(vocab):
        if k in tokenizer_vocab:
            continue
        tokenizer_vocab[k] = len(tokenizer_vocab)

    merges, missing_vocab = compute_merges(
        {k: v for k, v in tokenizer_vocab.items() if k not in added_vocab},
        byte_level_decoder
    )
    while add_missing_vocab and missing_vocab:
        for v in missing_vocab:
            tokenizer_vocab[v] = len(tokenizer_vocab)
        
        merges, missing_vocab = compute_merges(
            {k: v for k, v in tokenizer_vocab.items() if k not in added_vocab},
            byte_level_decoder
        )

    tokenizer_merges = tokenizer_json['model']['merges']
    merges_set = {tuple(m) for m in tokenizer_merges}
    
    new_merges = []
    for m in merges:
        if m in merges_set:
            continue
        new_merges.append(m)
        merges_set.add(m)

    tokenizer_merges += new_merges

    return tokenizer.__class__(
        tokenizer_object=Tokenizer.from_str(json.dumps(tokenizer_json)),
        **tokenizer.init_kwargs
    )


if __name__ == '__main__':
    fire.Fire(main)
