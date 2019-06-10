# data: combine3


# Done --- create checkpoint
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=pretrained_full_tune_combine2_exp --train_size=0
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=ckpt_pretrained_exp --subred_size=2 --train_size=0
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=ckpt_pretrained_exp --subred_size=2 --create_ckpt=True

# ------ BASELINE ------

# pre-trained
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=pretrained_combine2_exp --coverage=1 --convert_to_coverage_model=1
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=pretrained_combine2_exp --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# pre-trained + full tuning
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=pretrained_full_tune_combine2_exp --batch_size=16 --train_size=19050
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=pretrained_full_tune_combine2_exp --coverage=1 --convert_to_coverage_model=1
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=pretrained_full_tune_combine2_exp --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1


# ------ MULTI ATTENTION ------

# use_multi_attn
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=load_word_emb_multi_attn_combine2_exp --use_multi_attn=True --subred_size=2 --train_size=19050 # 2780
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=load_word_emb_multi_attn_combine2_exp --use_multi_attn=True --subred_size=2 --train_size=19050 --coverage=1 --convert_to_coverage_model=1
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=load_word_emb_multi_attn_combine2_exp --use_multi_attn=True --subred_size=2 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1


# use_multi_pgen
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=load_word_emb_multi_pgen_combine2_exp --use_multi_pgen=True --subred_size=2 --train_size=19050
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=load_word_emb_multi_pgen_combine2_exp --use_multi_pgen=True --subred_size=2 --train_size=19050 --coverage=1 --convert_to_coverage_model=1
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=load_word_emb_multi_pgen_combine2_exp --use_multi_pgen=True --subred_size=2 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1


# use_multi_pvocab
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=load_word_emb_multi_pvocab_combine2_exp --use_multi_pvocab=True --subred_size=2 --train_size=19050
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=load_word_emb_multi_pvocab_combine2_exp --use_multi_pvocab=True --subred_size=2 --train_size=19050 --coverage=1 --convert_to_coverage_model=1
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=load_word_emb_multi_pvocab_combine2_exp --use_multi_pvocab=True --subred_size=2 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=big_multi_pvocab_combine2_exp --use_multi_pvocab=True --subred_size=2 --hidden_dim=256 --emb_dim=128 --batch_size=4 --train_size=19050
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=big_multi_pvocab_combine2_exp --use_multi_pvocab=True --subred_size=2 --hidden_dim=256 --emb_dim=128 --batch_size=4 --train_size=19050 --coverage=1 --convert_to_coverage_model=1

# python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=big_multi_pvocab_combine2_exp --use_multi_pvocab=True --subred_size=2 --hidden_dim=256 --emb_dim=128 --batch_size=4 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=big_multi_pvocab_combine2_exp --use_multi_pvocab=True --subred_size=2 --hidden_dim=256 --emb_dim=128 --batch_size=4 --create_ckpt=True

# Done --- inital training
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=load_word_emb_multi_attn_combine2_exp --use_multi_attn=True --subred_size=2 --train_size=0
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=load_word_emb_multi_pgen_combine2_exp --use_multi_pgen=True --subred_size=2 --train_size=0
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=load_word_emb_multi_pvocab_combine2_exp --use_multi_pvocab=True --subred_size=2 --train_size=0

# Done ------ CREATE CHCEKPOINT ------

# # use_multi_attn, build graph: 131s -> 89s -> 46s, run train 0: 223s -> 119s -> 57s
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=ckpt_multi_attn_exp --use_multi_attn=True --subred_size=2 --train_size=0
# # use_multi_pgen, build graph: 131s -> 88s -> 45s, run train 0: 218s -> 119s -> 57s
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=ckpt_multi_pgen_exp --use_multi_pgen=True --subred_size=2 --train_size=0
# # use_multi_pvocab, build graph: 131s -> .. -> 45s, run train 0: -> 116s
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=ckpt_multi_pvocab_exp --use_multi_pvocab=True --subred_size=2 --train_size=0

# # use_multi_attn
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=ckpt_multi_attn_exp --use_multi_attn=True --subred_size=2 --create_ckpt=True
# # use_multi_pgen
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=ckpt_multi_pgen_exp --use_multi_pgen=True --subred_size=2 --create_ckpt=True 
# # use_multi_pvocab
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine2/finished_combine2_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=ckpt_multi_pvocab_exp --use_multi_pvocab=True --subred_size=2 --create_ckpt=True 