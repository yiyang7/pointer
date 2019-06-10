# # convert
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=pretrained_combine10_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --lr=0.15 --coverage=1 --convert_to_coverage_model=1
# # decode
# python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=pretrained_combine10_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --lr=0.15 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# --- fine_tune
# train
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_combine10_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --lr=0.15 --train_size=10000 --fine_tune=True

# cp -r log/fine_tune_combine10_exp log/fine_tune_combine10_exp_10k

# # convert
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_combine10_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --lr=0.15 --coverage=1 --convert_to_coverage_model=1
# # decode
# python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_combine10_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --lr=0.15 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# --- full_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_combine10_exp1 --hidden_dim=256 --emb_dim=128 --batch_size=16 --lr=0.15 --train_size=10000

cp -r log/full_tune_combine10_exp1 log/full_tune_combine10_exp_10k

# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_combine10_exp1 --hidden_dim=256 --emb_dim=128 --batch_size=16 --lr=0.15 --coverage=1 --convert_to_coverage_model=1
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_combine_all/finished_combine_all_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_combine10_exp1 --hidden_dim=256 --emb_dim=128 --batch_size=16 --lr=0.15 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1