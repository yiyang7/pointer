# # 1 data_path
# # 2 exp_name
# # relationships
# # --- full_tune
# # train
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_relationships/finished_relationships_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_relationships_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000
# # convert
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_relationships/finished_relationships_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_relationships_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1
# # decode
# python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_relationships/finished_relationships_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_relationships_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# # --- fine_tune
# # train
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_relationships/finished_relationships_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_relationships_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000 --fine_tune=True
# # convert
# python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_relationships/finished_relationships_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_relationships_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1 --fine_tune=True
# # decode
# python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_relationships/finished_relationships_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_relationships_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1 --fine_tune=True

# 1 data_path
# 2 exp_name
# legaladvice
# --- full_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_legaladvice/finished_legaladvice_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_legaladvice_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_legaladvice/finished_legaladvice_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_legaladvice_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_legaladvice/finished_legaladvice_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_legaladvice_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# --- fine_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_legaladvice/finished_legaladvice_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_legaladvice_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000 --fine_tune=True
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_legaladvice/finished_legaladvice_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_legaladvice_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1 --fine_tune=True
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_legaladvice/finished_legaladvice_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_legaladvice_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1 --fine_tune=True



# 1 data_path
# 2 exp_name
# nfl
# --- full_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_nfl/finished_nfl_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_nfl_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_nfl/finished_nfl_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_nfl_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_nfl/finished_nfl_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_nfl_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# # --- fine_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_nfl/finished_nfl_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_nfl_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000 --fine_tune=True
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_nfl/finished_nfl_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_nfl_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1 --fine_tune=True
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_nfl/finished_nfl_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_nfl_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1 --fine_tune=True



# 1 data_path
# 2 exp_name
# pettyrevenge
# --- full_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_pettyrevenge/finished_pettyrevenge_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_pettyrevenge_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_pettyrevenge/finished_pettyrevenge_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_pettyrevenge_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_pettyrevenge/finished_pettyrevenge_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_pettyrevenge_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# --- fine_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_pettyrevenge/finished_pettyrevenge_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_pettyrevenge_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000 --fine_tune=True
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_pettyrevenge/finished_pettyrevenge_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_pettyrevenge_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1 --fine_tune=True
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_pettyrevenge/finished_pettyrevenge_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_pettyrevenge_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1 --fine_tune=True



# 1 data_path
# 2 exp_name
# yiyang's original
# atheismbot
# --- full_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_atheismbot/finished_atheismbot_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_atheismbot_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_atheismbot/finished_atheismbot_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_atheismbot_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_atheismbot/finished_atheismbot_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_atheismbot_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# --- fine_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_atheismbot/finished_atheismbot_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_atheismbot_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000 --fine_tune=True
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_atheismbot/finished_atheismbot_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_atheismbot_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1 --fine_tune=True
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_atheismbot/finished_atheismbot_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_atheismbot_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1 --fine_tune=True


# 1 data_path
# 2 exp_name
# ShouldIbuythisgame
# --- full_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_ShouldIbuythisgame/finished_ShouldIbuythisgame_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_ShouldIbuythisgame_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_ShouldIbuythisgame/finished_ShouldIbuythisgame_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_ShouldIbuythisgame_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_ShouldIbuythisgame/finished_ShouldIbuythisgame_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_ShouldIbuythisgame_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# --- fine_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_ShouldIbuythisgame/finished_ShouldIbuythisgame_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_ShouldIbuythisgame_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000 --fine_tune=True
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_ShouldIbuythisgame/finished_ShouldIbuythisgame_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_ShouldIbuythisgame_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1 --fine_tune=True
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_ShouldIbuythisgame/finished_ShouldIbuythisgame_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_ShouldIbuythisgame_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1 --fine_tune=True


# 1 data_path
# 2 exp_name
# ukpolitics
# --- full_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_ukpolitics/finished_ukpolitics_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_ukpolitics_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_ukpolitics/finished_ukpolitics_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_ukpolitics_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_ukpolitics/finished_ukpolitics_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_ukpolitics_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# --- fine_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_ukpolitics/finished_ukpolitics_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_ukpolitics_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000 --fine_tune=True
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_ukpolitics/finished_ukpolitics_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_ukpolitics_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1 --fine_tune=True
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_ukpolitics/finished_ukpolitics_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_ukpolitics_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1 --fine_tune=True


# 1 data_path
# 2 exp_name
# Dogtraining
# --- full_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_Dogtraining/finished_Dogtraining_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_Dogtraining_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_Dogtraining/finished_Dogtraining_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_Dogtraining_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_Dogtraining/finished_Dogtraining_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_Dogtraining_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# --- fine_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_Dogtraining/finished_Dogtraining_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_Dogtraining_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000 --fine_tune=True
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_Dogtraining/finished_Dogtraining_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_Dogtraining_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1 --fine_tune=True
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_Dogtraining/finished_Dogtraining_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_Dogtraining_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1 --fine_tune=True


# 1 data_path
# 2 exp_name
# AskHistorians
# --- full_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_AskHistorians/finished_AskHistorians_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_AskHistorians_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_AskHistorians/finished_AskHistorians_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_AskHistorians_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_AskHistorians/finished_AskHistorians_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_AskHistorians_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# --- fine_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_AskHistorians/finished_AskHistorians_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_AskHistorians_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000 --fine_tune=True
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_AskHistorians/finished_AskHistorians_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_AskHistorians_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1 --fine_tune=True
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_AskHistorians/finished_AskHistorians_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_AskHistorians_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1 --fine_tune=True



# 1 data_path
# 2 exp_name
# Anxiety
# --- full_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_Anxiety/finished_Anxiety_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_Anxiety_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_Anxiety/finished_Anxiety_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_Anxiety_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_Anxiety/finished_Anxiety_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=full_tune_Anxiety_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1

# --- fine_tune
# train
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_Anxiety/finished_Anxiety_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_Anxiety_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --train_size=10000 --fine_tune=True
# convert
python run_summarization.py --mode=train --data_path=../processed_10_1k/processed_Anxiety/finished_Anxiety_story/chunked/train* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_Anxiety_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --coverage=1 --convert_to_coverage_model=1 --fine_tune=True
# decode
python run_summarization.py --mode=decode --data_path=../processed_10_1k/processed_Anxiety/finished_Anxiety_story/chunked/test* --vocab_path=../finished_cnn_files/vocab --log_root=log --exp_name=fine_tune_Anxiety_exp --hidden_dim=256 --emb_dim=128 --batch_size=16 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1 --fine_tune=True

